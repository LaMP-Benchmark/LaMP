from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForCausalLM
# from transformers.models.llama import LlamaTokenizer
from transformers.data.data_collator import DataCollatorForSeq2Seq
import argparse
from metrics.classification_metrics import create_metric_f1_accuracy, create_metric_mae_rmse
from metrics.generation_metrics import create_metric_bleu_rouge_meteor
from data.datasets import get_all_labels, GeneralSeq2SeqDataset, create_preprocessor, convert_to_hf_dataset
from prompts.prompts import create_prompt_generator
import json
import os

parser = argparse.ArgumentParser()

parser.add_argument("--validation_data", required = True)
parser.add_argument("--model_addr", required = True)
parser.add_argument("--task", required = True)
parser.add_argument("--output_dir", required = True)
parser.add_argument("--use_profile", action = "store_true")
parser.add_argument("--max_length", type = int, default = 256)
parser.add_argument("--generation_max_length", type = int, default = 128)
parser.add_argument("--per_device_batch_size", type = int, default = 16)
parser.add_argument("--generation_num_beams", type = int, default = 4)
parser.add_argument("--num_retrieved", type = int, default = 1)
parser.add_argument("--retriever", default = "bm25")
parser.add_argument("--is_ranked", action = "store_true")
parser.add_argument("--cache_dir", default = "./cache")


if __name__ == "__main__":

    opts = parser.parse_args()
    model = AutoModelForSeq2SeqLM.from_pretrained(opts.model_addr, cache_dir=opts.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(opts.model_addr, cache_dir=opts.cache_dir)
    collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model = model, max_length = opts.max_length)

    task = opts.task
    if opts.use_profile:
        prompt_generator, contriver = create_prompt_generator(opts.num_retrieved, opts.retriever, opts.is_ranked, opts.max_length, tokenizer)
    else:
        prompt_generator, contriver = None, None

    if task == "LaMP-1":
        labels = get_all_labels(task)
        eval_dataset = GeneralSeq2SeqDataset(opts.validation_data, opts.use_profile, task, prompt_generator)
        compute_metrics = create_metric_f1_accuracy(tokenizer = tokenizer, all_labels = labels)
    elif task == "LaMP-2":
        labels = get_all_labels(task)
        eval_dataset = GeneralSeq2SeqDataset(opts.validation_data, opts.use_profile, task, prompt_generator)
        compute_metrics = create_metric_f1_accuracy(tokenizer = tokenizer, all_labels = labels)
    elif task == "LaMP-3":
        labels = get_all_labels(task)
        eval_dataset = GeneralSeq2SeqDataset(opts.validation_data, opts.use_profile, task, prompt_generator)
        compute_metrics = create_metric_mae_rmse(tokenizer = tokenizer, all_labels = labels)
    elif task == "LaMP-4":
        eval_dataset = GeneralSeq2SeqDataset(opts.validation_data, opts.use_profile, task, prompt_generator)
        compute_metrics = create_metric_bleu_rouge_meteor(tokenizer = tokenizer)
    elif task == "LaMP-5":
        eval_dataset = GeneralSeq2SeqDataset(opts.validation_data, opts.use_profile, task, prompt_generator)
        compute_metrics = create_metric_bleu_rouge_meteor(tokenizer = tokenizer)
    elif task == "LaMP-7":
        eval_dataset = GeneralSeq2SeqDataset(opts.validation_data, opts.use_profile, task, prompt_generator)
        compute_metrics = create_metric_bleu_rouge_meteor(tokenizer = tokenizer)
    elif task == "LaMP-6":
        eval_dataset = GeneralSeq2SeqDataset(opts.validation_data, opts.use_profile, task, prompt_generator)
        compute_metrics = create_metric_bleu_rouge_meteor(tokenizer = tokenizer)
   
    eval_dataset = convert_to_hf_dataset(eval_dataset, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = opts.max_length), batched=True)

    if contriver:
        contriver = contriver.to("cpu")

    training_args = Seq2SeqTrainingArguments(
        output_dir = opts.output_dir,
        do_eval = True,
        per_device_eval_batch_size = opts.per_device_batch_size,
        generation_num_beams = opts.generation_num_beams,
        predict_with_generate = True,
        eval_accumulation_steps = 1,
        generation_max_length = opts.generation_max_length
    )

    trainer = Seq2SeqTrainer(
        model = model,
        args = training_args,
        data_collator = collator,
        eval_dataset = eval_dataset,
        tokenizer = tokenizer,
        compute_metrics = compute_metrics
    )
    results = trainer.evaluate(eval_dataset)
    print(results)

    with open(os.path.join(opts.output_dir,'results_output.json'), 'w') as file:
        json.dump(results, file, indent = 4)