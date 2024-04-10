from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.data.data_collator import DataCollatorForSeq2Seq
import argparse
from metrics.classification_metrics import create_metric_f1_accuracy, create_metric_mae_rmse
from metrics.generation_metrics import create_metric_bleu_rouge_meteor
from data.datasets import get_all_labels, GeneralSeq2SeqDataset, create_preprocessor, convert_to_hf_dataset
from prompts.prompts import create_prompt_generator
import os
import json

parser = argparse.ArgumentParser()

parser.add_argument("--train_data", required = True)
parser.add_argument("--validation_data", required = True)
parser.add_argument("--test_data", default="")
parser.add_argument("--model_name", required = True)
parser.add_argument("--task", required = True)
parser.add_argument("--output_dir", required = True)
parser.add_argument("--retriever", default = "bm25")
parser.add_argument("--use_profile", action = "store_true")
parser.add_argument("--is_ranked", action = "store_true")
parser.add_argument("--max_length", type = int, default = 256)
parser.add_argument("--generation_max_length", type = int, default = 128)
parser.add_argument("--per_device_batch_size", type = int, default = 16)
parser.add_argument("--learning_rate", type = float, default = 5e-5)
parser.add_argument("--weight_decay", type = float, default = 0.0001)
parser.add_argument("--num_train_epochs", type = int, default = 10)
parser.add_argument("--lr_scheduler_type", default = "linear")
parser.add_argument("--warmup_ratio", type = float, default = 0.05)
parser.add_argument("--generation_num_beams", type = int, default = 4)
parser.add_argument("--num_retrieved", type = int, required=True)
parser.add_argument("--gradient_accumulation_steps", type = int, default = 1)
parser.add_argument("--cache_dir", default = "./cache")


if __name__ == "__main__":

    opts = parser.parse_args()
    
    model = AutoModelForSeq2SeqLM.from_pretrained(opts.model_name, cache_dir=opts.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(opts.model_name, cache_dir=opts.cache_dir)
    collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model = model, max_length = opts.max_length)

    task = opts.task
    if opts.use_profile:
        prompt_generator, contriver = create_prompt_generator(opts.num_retrieved, opts.retriever, opts.is_ranked, opts.max_length, tokenizer)
    else:
        prompt_generator, contriver = None, None

    greater_is_better = True
    if task == "LaMP-1":
        train_dataset, labels = GeneralSeq2SeqDataset(opts.train_data, opts.use_profile, task, prompt_generator), get_all_labels(task)
        eval_dataset = GeneralSeq2SeqDataset(opts.validation_data, opts.use_profile, task, prompt_generator)
        if opts.test_data:
            test_dataset = GeneralSeq2SeqDataset(opts.test_data, opts.use_profile, task, prompt_generator)
        compute_metrics = create_metric_f1_accuracy(tokenizer = tokenizer, all_labels = labels)
        best_metric = "accuracy"
    elif task == "LaMP-2-old":
        train_dataset, labels = GeneralSeq2SeqDataset(opts.train_data, opts.use_profile, task, prompt_generator), get_all_labels(task)
        eval_dataset = GeneralSeq2SeqDataset(opts.validation_data, opts.use_profile, task, prompt_generator)
        if opts.test_data:
            test_dataset = GeneralSeq2SeqDataset(opts.test_data, opts.use_profile, task, prompt_generator)
        compute_metrics = create_metric_f1_accuracy(tokenizer = tokenizer, all_labels = labels)
        best_metric = "accuracy"
    elif task == "LaMP-2":
        train_dataset, labels = GeneralSeq2SeqDataset(opts.train_data, opts.use_profile, task, prompt_generator), get_all_labels(task)
        eval_dataset = GeneralSeq2SeqDataset(opts.validation_data, opts.use_profile, task, prompt_generator)
        if opts.test_data:
            test_dataset = GeneralSeq2SeqDataset(opts.test_data, opts.use_profile, task, prompt_generator)
        compute_metrics = create_metric_f1_accuracy(tokenizer = tokenizer, all_labels = labels)
        best_metric = "accuracy"
    elif task == "LaMP-3":
        train_dataset, labels = GeneralSeq2SeqDataset(opts.train_data, opts.use_profile, task, prompt_generator), get_all_labels(task)
        eval_dataset = GeneralSeq2SeqDataset(opts.validation_data, opts.use_profile, task, prompt_generator)
        if opts.test_data:
            test_dataset = GeneralSeq2SeqDataset(opts.test_data, opts.use_profile, task, prompt_generator)
        compute_metrics = create_metric_mae_rmse(tokenizer = tokenizer, all_labels = labels)
        best_metric = "mae"
        greater_is_better = False
    elif task == "LaMP-4":
        train_dataset = GeneralSeq2SeqDataset(opts.train_data, opts.use_profile, task, prompt_generator)
        eval_dataset = GeneralSeq2SeqDataset(opts.validation_data, opts.use_profile, task, prompt_generator)
        if opts.test_data:
            test_dataset = GeneralSeq2SeqDataset(opts.test_data, opts.use_profile, task, prompt_generator)
        compute_metrics = create_metric_bleu_rouge_meteor(tokenizer = tokenizer)
        best_metric = "rouge-1"
    elif task == "LaMP-5":
        train_dataset = GeneralSeq2SeqDataset(opts.train_data, opts.use_profile, task, prompt_generator)
        eval_dataset = GeneralSeq2SeqDataset(opts.validation_data, opts.use_profile, task, prompt_generator)
        if opts.test_data:
            test_dataset = GeneralSeq2SeqDataset(opts.test_data, opts.use_profile, task, prompt_generator)
        compute_metrics = create_metric_bleu_rouge_meteor(tokenizer = tokenizer)
        best_metric = "rouge-1"
    elif task == "LaMP-7":
        train_dataset = GeneralSeq2SeqDataset(opts.train_data, opts.use_profile, task, prompt_generator)
        eval_dataset = GeneralSeq2SeqDataset(opts.validation_data, opts.use_profile, task, prompt_generator)
        if opts.test_data:
            test_dataset = GeneralSeq2SeqDataset(opts.test_data, opts.use_profile, task, prompt_generator)
        compute_metrics = create_metric_bleu_rouge_meteor(tokenizer = tokenizer)
        best_metric = "rouge-1"
    elif task == "LaMP-6":
        train_dataset = GeneralSeq2SeqDataset(opts.train_data, opts.use_profile, task, prompt_generator)
        eval_dataset = GeneralSeq2SeqDataset(opts.validation_data, opts.use_profile, task, prompt_generator)
        if opts.test_data:
            test_dataset = GeneralSeq2SeqDataset(opts.test_data, opts.use_profile, task, prompt_generator)
        compute_metrics = create_metric_bleu_rouge_meteor(tokenizer = tokenizer)
        best_metric = "rouge-1"
    
    train_dataset = convert_to_hf_dataset(train_dataset, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = opts.max_length), batched=True)
    eval_dataset = convert_to_hf_dataset(eval_dataset, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = opts.max_length), batched=True)
    if opts.test_data:
        test_dataset = convert_to_hf_dataset(test_dataset, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = opts.max_length), batched=True)

    if contriver:
        contriver = contriver.to("cpu")

    training_args = Seq2SeqTrainingArguments(
        output_dir = opts.output_dir,
        do_train = True,
        do_eval = True,
        evaluation_strategy = "epoch",
        per_device_train_batch_size = opts.per_device_batch_size,
        per_device_eval_batch_size = opts.per_device_batch_size,
        gradient_accumulation_steps = opts.gradient_accumulation_steps,
        learning_rate = opts.learning_rate,
        weight_decay = opts.weight_decay,
        num_train_epochs = opts.num_train_epochs,
        lr_scheduler_type = opts.lr_scheduler_type,
        warmup_ratio = opts.warmup_ratio,
        generation_num_beams = opts.generation_num_beams,
        predict_with_generate = True,
        save_strategy = "epoch",
        logging_steps = 50,
        eval_accumulation_steps = 1,
        generation_max_length = opts.generation_max_length,
        load_best_model_at_end = True,
        metric_for_best_model = best_metric,
        greater_is_better = greater_is_better
    )

    trainer = Seq2SeqTrainer(
        model = model,
        args = training_args,
        data_collator = collator,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        tokenizer = tokenizer,
        compute_metrics = compute_metrics
    )

    trainer.train()

    if opts.test_data:
        results = trainer.evaluate(test_dataset)
        print(results)

        with open(os.join(opts.output_dir,'results_output.json'), 'w') as file:
            json.dump(results, file, indent = 4)