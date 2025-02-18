import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.data.data_collator import DataCollatorForSeq2Seq
from data.datasets import create_per_user_dataset, create_preprocessor
import json
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import os
import torch


def is_directory_empty(path):
    if os.path.exists(path):
        if not os.listdir(path):
            return True
        else:
            return False
    else:
        return True


parser = argparse.ArgumentParser()

parser.add_argument("--train_data", required = True)
parser.add_argument("--user_ids", default="")
parser.add_argument("--task", required = True)
parser.add_argument("--output_dir", required = True)
parser.add_argument("--max_length", type = int, default = 512)
parser.add_argument("--num_shards", type = int, default = 1)
parser.add_argument("--shard_id", type = int, default = 0)
parser.add_argument("--generation_max_length", type = int, default = 512)
parser.add_argument("--per_device_batch_size", type = int, default = 16)
parser.add_argument("--learning_rate", type = float, default = 5e-5)
parser.add_argument("--weight_decay", type = float, default = 0.0001)
parser.add_argument("--num_train_epochs", type = int, default = 30)
parser.add_argument("--lora_r", type = int, default = 8)
parser.add_argument("--lr_scheduler_type", default = "linear")
parser.add_argument("--warmup_ratio", type = float, default = 0.05)
parser.add_argument("--gradient_accumulation_steps", type = int, default = 1)
parser.add_argument("--cache", default="./cache")



if __name__ == "__main__":

    opts = parser.parse_args()
    print(opts)
    if opts.user_ids:
        with open(opts.user_ids) as file:
            all_user_ids = [str(x) for x in json.load(file)]
            shard_size = len(all_user_ids) // opts.num_shards + 1
            user_ids = all_user_ids[int(opts.shard_id * shard_size):int((opts.shard_id + 1) * shard_size)]
    else:
        user_ids = None
    user_datasets = create_per_user_dataset(opts.train_data, user_ids, opts.task, opts.cache)

    model_name_or_path = "google/flan-t5-xxl"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, cache_dir = opts.cache)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir = opts.cache)  
    processor = create_preprocessor(tokenizer = tokenizer, max_length = opts.max_length)
    collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model = model, max_length = opts.max_length)
    for key, dataset in user_datasets.items():
        print(key)
        if not is_directory_empty(os.path.join(opts.output_dir, 'adaptors', key)):
            continue
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=opts.lora_r, lora_alpha=32, lora_dropout=0.1, target_modules=['q','v','k']
        )
        model = get_peft_model(model, peft_config)
        encoded_dataset = dataset.map(processor, batched=True)
        training_args = Seq2SeqTrainingArguments(
            output_dir = os.path.join(opts.output_dir, 'adaptors', key),
            do_train = True,
            do_eval = False,
            per_device_train_batch_size = opts.per_device_batch_size,
            gradient_accumulation_steps = opts.gradient_accumulation_steps,
            learning_rate = opts.learning_rate,
            weight_decay = opts.weight_decay,
            num_train_epochs = opts.num_train_epochs,
            lr_scheduler_type = opts.lr_scheduler_type,
            warmup_ratio = opts.warmup_ratio,
            save_strategy = "epoch",
            save_total_limit=1,
            logging_steps = 10,
            generation_max_length = opts.generation_max_length,
            save_only_model = True
        )

        trainer = Seq2SeqTrainer(
            model = model,
            args = training_args,
            data_collator = collator,
            train_dataset = encoded_dataset,
            tokenizer = tokenizer
        )

        trainer.train()
        model.unload()
        trainer = None
        torch.cuda.empty_cache()
        
        