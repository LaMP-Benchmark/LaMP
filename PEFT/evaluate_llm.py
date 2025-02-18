import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.data.data_collator import DataCollatorForSeq2Seq
from data.datasets import create_per_user_dataset_test, create_preprocessor
import json
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import os
import torch
import glob
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument("--test_data", required = True)
parser.add_argument("--user_ids", default = "")
parser.add_argument("--golds_addr", required = True)
parser.add_argument("--task", required = True)
parser.add_argument("--user_checkpoints", required = True)
parser.add_argument("--output_dir", required = True)
parser.add_argument("--max_length", type = int, default = 512)
parser.add_argument("--num_shards", type = int, default = 1)
parser.add_argument("--shard_id", type = int, default = 0)
parser.add_argument("--generation_max_length", type = int, default = 128)
parser.add_argument("--per_device_batch_size", type = int, default = 16)
parser.add_argument("--generation_num_beams", type = int, default = 4)
parser.add_argument("--cache", default="./cache")

if __name__ == "__main__":
    opts = parser.parse_args()
    print(opts)
    with open(opts.user_ids) as file:
        all_user_ids = [str(x) for x in json.load(file)]
        shard_size = len(all_user_ids) // opts.num_shards + 1
        user_ids = all_user_ids[int(opts.shard_id * shard_size):int((opts.shard_id + 1) * shard_size)]
    
    user_datasets = create_per_user_dataset_test(opts.test_data, user_ids, opts.task, opts.cache)

    model_name_or_path = "google/flan-t5-xxl"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, cache_dir = opts.cache)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir = opts.cache)  
    processor = create_preprocessor(tokenizer = tokenizer, max_length = opts.max_length)
    collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model = model, max_length = opts.max_length)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=['q','v','k']
    )
    model = get_peft_model(model, peft_config)

    final_outputs = []

    for key, dataset in user_datasets.items():
        model.unload()
        checkpoitns = glob.glob(os.path.join(opts.user_checkpoints, 'adaptors', key, '*'))
        if len(checkpoitns) > 0:
            checkpoint_addr = checkpoitns[0]
            print(checkpoint_addr)
            model.load_adapter(checkpoint_addr, key)
            model.set_adapter(key)
        encoded_dataset = dataset.map(processor, batched=True)
        
        training_args = Seq2SeqTrainingArguments(
            output_dir = opts.output_dir,
            do_train = False,
            do_eval = True,
            per_device_train_batch_size = opts.per_device_batch_size,
            generation_max_length = opts.generation_max_length,
            generation_num_beams = opts.generation_num_beams,
            predict_with_generate=True,
            eval_accumulation_steps = 1
        )

        trainer = Seq2SeqTrainer(
            model = model,
            args = training_args,
            data_collator = collator,
            train_dataset = encoded_dataset,
            tokenizer = tokenizer
        )

        preds = trainer.predict(encoded_dataset).predictions
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        preds = tokenizer.batch_decode(preds, skip_special_tokens = True)
        for data, pred in zip(dataset, preds):
            final_outputs.append(
                {
                    "id" : data['id'],
                    "output" : pred
                }
            )
    prediction_addr = os.path.join(opts.output_dir, 'predictions.json')

    with open(prediction_addr, 'w') as file:
        json.dump(
            {
                "task" : opts.task,
                "golds" : final_outputs
            },
            file,
            indent=4
        )