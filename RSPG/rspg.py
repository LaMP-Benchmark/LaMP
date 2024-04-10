import argparse
import torch
import json
import os
from pathlib import Path
from utils.log import init_logger
from pathlib import Path
from utils.distributed import init_distributed_mode, init_signal_handler
import torch
from modeling import optim
from data.dataset import RSPGDataset
from data.collators import RSPGPostCollator, RSPGPreCollator
from modeling.modeling import RSPG, Trainer
from modeling.utils import load_checkpoint, average_main, save_checkpoint
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import tqdm
from transformers import AutoTokenizer, AutoModel, PretrainedConfig
from metrics.evaluation import LaMPEvaluation
import numpy as np
import glob


parser = argparse.ArgumentParser()

parser.add_argument("--train_data", required = True, help="the training data")
parser.add_argument("--val_data", required = True, help="the validation data")
parser.add_argument("--rspg_type", required = True, help="RSPG type: [Pre, Post]")

parser.add_argument("--val_lamp_golds", required = True, help="the validation data")
parser.add_argument("--do_filtering", action='store_true')

parser.add_argument("--task", required = True, help="task")
parser.add_argument("--do_train", action='store_true', help="perform training")
parser.add_argument("--do_validation", action='store_true', help="perform validation")
parser.add_argument("--max_length_input", type = int, default = 512, help="maximum input length")
parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/', help='models are saved here')
parser.add_argument('--model_path', type=str, default='none', help='path for a checkpoint to start training from that')
parser.add_argument("--per_gpu_batch_size", default=1, type=int, 
           help="Batch size per GPU/CPU for training.")
parser.add_argument("--local-rank", type=int, default=-1,
           help="For distributed training: local_rank")
parser.add_argument("--main_port", type=int, default=-1,
           help="Main port (for multi-node SLURM jobs)")
parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")
parser.add_argument('--eval_freq', type=int, default=500,
           help='evaluate model every <eval_freq> steps during training')
parser.add_argument('--save_freq', type=int, default=5000,
           help='save model every <save_freq> steps during training')
parser.add_argument('--eval_print_freq', type=int, default=1000,
           help='print intermdiate results of evaluation every <eval_print_freq> steps')
parser.add_argument('--warmup_steps', type=int, default=1000, help="number of warmup steps")
parser.add_argument('--total_steps', type=int, default=1000, help="number of training steps")
parser.add_argument('--scheduler_steps', type=int, default=None, 
           help='total number of step for the scheduler, if None then scheduler_total_step = total_step')
parser.add_argument('--accumulation_steps', type=int, default=1, help="number of gradient accumulation steps")
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--clip', type=float, default=1., help='gradient clipping')
parser.add_argument('--optim', type=str, default='adam', help="optimizer which is used for training")
parser.add_argument('--scheduler', type=str, default='fixed', help="scheduler which is used for training")
parser.add_argument('--weight_decay', type=float, default=0.0, help="weight decay rate")
parser.add_argument('--temperature', type=float, default=1.0, help="distillation temperature")
parser.add_argument('--fixed_lr', action='store_true', help="use a fixed lr")


def train(opts, model, optimizer, scheduler, step, dataset, collator, checkpoint_path, test_dataset, logger, compute_metrics):
    
    if opts.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opts.checkpoint_dir)/opts.name)
        except:
           tb_logger = None
           logger.warning('Tensorboard is not available.')

    torch.manual_seed(opts.global_rank + opts.seed) #different seed for different sampling depending on global_rank
    train_sampler = DistributedSampler(dataset, num_replicas=opts.n_gpu_per_node, rank=opts.local_rank)
    train_dataloader = DataLoader(
        dataset,
        sampler = train_sampler,
        batch_size = opts.per_gpu_batch_size,
        drop_last = True,
        num_workers = 10,
        collate_fn = collator
    )

    loss, curr_loss = 0.0, 0.0
    epoch = 1
    model.train()
    bar = tqdm.tqdm(total=opts.total_steps)
    temp_step = 0
    while step < opts.total_steps:
        epoch += 1
        for i, batch in enumerate(train_dataloader):
            temp_step += 1
            train_loss = model(
                input_ids = batch['input_ids'].cuda(),
                attention_mask = batch['attention_mask'].cuda(),
                labels = batch['labels'].cuda()
            )[0]
            
            train_loss.backward()

            if temp_step % opts.accumulation_steps == 0:
                step += 1
                temp_step = 0
                bar.update(1)
                torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            train_loss = average_main(train_loss, opts)
            curr_loss += train_loss.item()
            
            if opts.is_main and step % opts.eval_freq == 0 and temp_step == 0 and step != 0:
                metrics = evaluate(model.module, test_dataset, collator, opts, step, logger, compute_metrics)
                if opts.is_main:
                    log = f"{step} / {opts.total_steps} |"
                    log += f"train: {curr_loss/opts.eval_freq:.4f} | {metrics}"
                    logger.info(log)
                    if tb_logger is not None:
                        tb_logger.add_scalar("Training", curr_loss / (opts.eval_freq), step)
                    curr_loss = 0.
                    save_checkpoint(model.module.model, optimizer, scheduler, step, opts, checkpoint_path, f"step-{step}")
                    model.train()
            if opts.is_main and step % opts.save_freq == 0 and temp_step == 0:
                save_checkpoint(model.module.model, optimizer, scheduler, step, opts, checkpoint_path, f"step-{step}")
            if step > opts.total_steps:
                break
    save_checkpoint(model.module.model, optimizer, scheduler, step, opts, checkpoint_path, f"step-{step}")

def evaluate(model, dataset, collator, opt, step, logger, evaluator, test_eval = False):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=10,
        collate_fn=collator
    )
    model.eval()
    with torch.no_grad():
        preds = []
        golds = []
        ids = []
        indices = []
        
        logger.info("Evaluation Started")
        for i, batch in enumerate(tqdm.tqdm(dataloader)):
            if test_eval:
                outputs = model(
                    input_ids=batch['input_ids'].cuda(), 
                    attention_mask = batch['attention_mask'].cuda(),
                )
            else:
                outputs = model.model(
                    input_ids=batch['input_ids'].cuda(), 
                    attention_mask = batch['attention_mask'].cuda(),
                )
            indices_max = torch.argmax(outputs, dim = -1)
            ids.extend(batch['id'])
            preds.extend([z[k] for k, z in zip(indices_max, batch['outputs'])])
            golds.extend(batch['gold'])
            indices.extend(indices_max.tolist())
    
    checkpoint_path = Path(opts.checkpoint_dir) / opts.name / "predictions" / str(step)
    checkpoint_path.mkdir(parents = True, exist_ok = True)
    
    with open(os.path.join(checkpoint_path, f'{str(opts.local_rank)}.json'), "w") as file:
        json.dump({"preds" : preds, "golds" : golds, "ids" : ids, "indices" : indices}, file)

    if opts.is_main:
        results = glob.glob(os.path.join(checkpoint_path, f'*.json'))
        preds, golds, ids, indices = [], [], [], []
        for addr in results:
            with open(addr) as file:
                temp = json.load(file)
                preds.extend(temp['preds'])
                golds.extend(temp['golds'])
                ids.extend(temp['ids'])
                indices.extend(temp['indices'])
        final_preds = {
            "task" : opts.task.replace("-", "_"),
            "golds" : [{"id" : id, "output" : out, "index":ind} for id, out, ind in zip(ids, preds, indices)]
        }
        final_preds_addr = os.path.join(checkpoint_path, f'final_preds.json')
        with open(final_preds_addr, "w") as file:
            json.dump(final_preds, file, indent=4)
        return evaluator.evaluate_task(final_preds_addr, opts.task.replace("-", "_"))

if __name__ == "__main__":
    opts = parser.parse_args()

    torch.manual_seed(opts.seed)
    init_distributed_mode(opts)
    init_signal_handler()

    checkpoint_path = Path(opts.checkpoint_dir) / opts.name
    checkpoint_exists = checkpoint_path.exists()
    
    if opts.is_distributed:
        torch.distributed.barrier()
    
    checkpoint_path.mkdir(parents = True, exist_ok = True)


    logger = init_logger(
        opts.is_main,
        opts.is_distributed,
        checkpoint_path / 'run.log'
    )

    logger.info(opts)

    tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-base-4096')

    task = opts.task
    if task == "LaMP-3":
        smaller_is_better = True
    else:
        smaller_is_better = False
    train_dataset = RSPGDataset(opts.train_data, smaller_is_better)
    val_dataset = RSPGDataset(opts.val_data, smaller_is_better)
    if opts.rspg_type == "Post":
        collator = RSPGPostCollator(tokenizer, opts.max_length_input)
    else:
        collator = RSPGPreCollator(tokenizer, opts.max_length_input)
    
    compute_metrics = LaMPEvaluation(
        single_gold_json_file_addr=opts.val_lamp_golds
    )

    if checkpoint_exists and opts.do_train:
        model, optimizer, scheduler, checkpoint_opts, step = load_checkpoint(RSPG, os.path.join(checkpoint_path, "checkpoint", "latest"), opts)
    elif opts.do_train:
        model = AutoModel.from_pretrained('allenai/longformer-base-4096')
        model.config.num_labels = 6
        model.config.init_model = 'allenai/longformer-base-4096'
        model = RSPG(model.config)
        model = Trainer(model, opts.temperature)
        optimizer, scheduler = optim.set_optim(opts, model)
        step = 0
    elif opts.do_validation:
        config = PretrainedConfig.from_pretrained(opts.model_path)
        model = RSPG.from_pretrained(opts.model_path, config = config)
    
    model = model.to(opts.local_rank)
    
    if opts.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opts.local_rank],
            output_device=opts.local_rank,
            find_unused_parameters=True,
        )
    
    if opts.do_train:
        train(opts, model, optimizer, scheduler, step, train_dataset, collator, checkpoint_path, val_dataset, logger, compute_metrics)
    
    if opts.do_validation and opts.is_main:
        metrics = evaluate(model, val_dataset, collator, opts, "validation", logger, compute_metrics, True)
        if opts.is_main:
            log = f"test: {metrics}"
            logger.info(log)

