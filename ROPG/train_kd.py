from pathlib import Path
from utils.distributed import init_distributed_mode, init_signal_handler
import torch
import numpy as np
import os
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler, RandomSampler
import tqdm
import argparse
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM
from data.collators import ReaderToRetreieverCollator
from data.datasets import ReaderToRetrieverDataset, get_all_labels
from prompts.prompts import create_query_corpus_generator
from trainers.trainer import KDReaderToRetrieverTrainer
from models.retriever import Contriever
from utils.util import average_main
from utils.log import init_logger
from models.optim import set_optim
from utils.util import save_checkpoint, load_checkpoint


def train(opts, model, optimizer, scheduler, step, dataset, collator, checkpoint_path):
    
    if opts.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opts.checkpoint_dir)/opts.name)
        except:
           tb_logger = None
           logger.warning('Tensorboard is not available.')

    torch.manual_seed(opts.global_rank + opts.seed)
    if opts.is_distributed:
        train_sampler = DistributedSampler(dataset, num_replicas=opts.n_gpu_per_node, rank=opts.local_rank)
    else:
        train_sampler = RandomSampler(dataset)
    bar = tqdm.tqdm(total=opts.total_steps)
    train_dataloader = DataLoader(
        dataset,
        sampler = train_sampler,
        batch_size = opts.per_gpu_batch_size,
        drop_last = True,
        num_workers = 10,
        collate_fn = collator,
    )

    loss, curr_loss = 0.0, 0.0
    epoch = 1
    model.train()
    temp_step = 0
    while step < opts.total_steps:
        epoch += 1
        for i, batch in enumerate(train_dataloader):
            temp_step += 1
            batch = {k:v.cuda() if type(v) != list else v for k, v in batch.items()}
            train_loss, scores, gold_scores = model(**batch)

            train_loss.backward()
            if temp_step % opts.accumulation_steps == 0:
                step += 1
                temp_step = 0
                torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                if opts.is_main:
                    bar.update(1)

            train_loss = average_main(train_loss, opts)
            curr_loss += train_loss.item()
            if opts.is_main and step % opts.save_freq == 0:
                save_checkpoint(model, optimizer, scheduler, step, opts, checkpoint_path, f"step-{step}")
            if step > opts.total_steps:
                save_checkpoint(model, optimizer, scheduler, step, opts, checkpoint_path, f"step-{step}")
                break
    
parser = argparse.ArgumentParser()

parser.add_argument("--train_data", required = True, help="training data")
parser.add_argument("--do_train", action='store_true', help="perform training")
parser.add_argument("--scores_path", required=True, help="address to pre-computed profile item score")

parser.add_argument("--max_length_query", type = int, default = 512, help="max length query")
parser.add_argument("--max_length_document", type = int, default = 512, help="max length document")

parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/', help='models are saved here')

parser.add_argument("--per_gpu_batch_size", default=1, type=int, 
           help="Batch size per GPU/CPU for training.")
parser.add_argument("--local-rank", type=int, default=-1,
           help="For distributed training: local_rank")
parser.add_argument("--main_port", type=int, default=-1,
           help="Main port (for multi-node SLURM jobs)")
parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")
parser.add_argument('--save_freq', type=int, default=5000,
           help='save model every <save_freq> steps during training')
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
parser.add_argument('--fixed_lr', action='store_true', help="use a fixed lr")

parser.add_argument('--ctx_size', type=int, default=20, help="number of docs per query for training")

parser.add_argument("--task", required = True, help="task name")
parser.add_argument("--model_path", default="", help="address to a checkpoint to be load")

parser.add_argument('--temperature', type=float, default=1.0, help="temperature for distillation")
parser.add_argument('--cache_dir', default="cache")






if __name__ == "__main__":
    opts = parser.parse_args()

    torch.manual_seed(opts.seed)
    init_distributed_mode(opts)
    init_signal_handler()

    checkpoint_path = Path(opts.checkpoint_dir)/opts.name
    checkpoint_exists = checkpoint_path.exists()
    
    if opts.is_distributed:
        torch.distributed.barrier()
    
    checkpoint_path.mkdir(parents = True, exist_ok = True)
    opts.output_dir = checkpoint_path


    logger = init_logger(
        opts.is_main,
        opts.is_distributed,
        checkpoint_path / 'run.log'
    )

    logger.info(opts)

    model = Contriever.from_pretrained('facebook/contriever', cache_dir = opts.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever', cache_dir = opts.cache_dir)
    collator = ReaderToRetreieverCollator(tokenizer = tokenizer, query_max_lenght = opts.max_length_query, document_max_length = opts.max_length_document, number_of_ctx = opts.ctx_size, scores_addr = opts.scores_path)
    
    task = opts.task
    
    reader_tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base', cache_dir = opts.cache_dir)
    query_corpus_generator = create_query_corpus_generator(task)

    greater_is_better = True

    if task == "LaMP-1":
        train_dataset, labels = ReaderToRetrieverDataset(opts.train_data, task, query_corpus_generator), get_all_labels(task)
        best_metric_generation = "accuracy"
    elif task == "LaMP-2":
        train_dataset, labels = ReaderToRetrieverDataset(opts.train_data, task, query_corpus_generator), get_all_labels(task)
        best_metric_generation = "accuracy"
    elif task == "LaMP-3":
        train_dataset, labels = ReaderToRetrieverDataset(opts.train_data, task, query_corpus_generator), get_all_labels(task)
        best_metric_generation = "mae"
        greater_is_better = False
    elif task == "LaMP-4":
        train_dataset = ReaderToRetrieverDataset(opts.train_data, task, query_corpus_generator)
        best_metric_generation = "rouge-1"
    elif task == "LaMP-5":
        train_dataset = ReaderToRetrieverDataset(opts.train_data, task, query_corpus_generator)
        best_metric_generation = "rouge-1"
    elif task == "LaMP-7":
        train_dataset = ReaderToRetrieverDataset(opts.train_data, task, query_corpus_generator)
        best_metric_generation = "rouge-1"
    elif task == "LaMP-6":
        train_dataset = ReaderToRetrieverDataset(opts.train_data, task, query_corpus_generator)
        best_metric_generation = "rouge-1"
    
    opts.greater_is_better = greater_is_better
    opts.reader_gold_metric = best_metric_generation

    if not checkpoint_exists and not opts.model_path:
        model = KDReaderToRetrieverTrainer(model = model, args = opts)
        model = model.to(opts.local_rank)
        optimizer, scheduler = set_optim(opts, model)
        step = 0
    elif checkpoint_exists and opts.model_path and opts.do_train:
        model, optimizer, scheduler, opt_checkpoint, step = load_checkpoint(Contriever, opts.model_path, opts)
        model = KDReaderToRetrieverTrainer(model = model, args = opts)
    
    if opts.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opts.local_rank],
            output_device=opts.local_rank,
            find_unused_parameters=True,
        )

    if opts.do_train:
        train(opts, model, optimizer, scheduler, step, train_dataset, collator, checkpoint_path)