from torch.utils.data import Dataset
import json
import datasets
import torch
import random
from itertools import combinations

def sublists_between_2_and_k(lst, k):
    sublists = []
    for size in range(2, k+1):  # Iterate through sizes from 2 to k
        for comb in combinations(lst, size):
            sublists.append(list(comb))
    return sublists

def sample_sublists(lst, k, num_samples):
    sublists = []
    for i in range(k+1, len(lst)):
        sub = list(random.sample(lst[:i], k-1))
        sub.sort(key=lambda x: x['date'])
        sub += [lst[i]]
        sublists.append(sub)
    while len(sublists) < num_samples:
        idx = random.randint(k+1, len(lst) - 1)
        sub = list(random.sample(lst[:idx], k-1))
        sub.sort(key=lambda x: x['date'])
        sub += [lst[idx]]
        sublists.append(sub)
    return sublists

def get_all_labels(task):
    if task == "classification_citation":
        return ["[1]","[2]"]
    elif task == "classification_news":
        return ["food & drink", "sports", "education", "parents", "religion", "travel", "business", "crime", "science & technology", "culture & arts", "entertainment", "politics", "women", "style & beauty", "healthy living"]
    elif task == "classification_movies":
        return ['sci-fi', 'based on a book', 'comedy', 'action', 'twist ending', 'dystopia', 'dark comedy', 'classic', 'psychology', 'fantasy', 'romance', 'thought-provoking', 'social commentary', 'violence', 'true story']
    elif task == "classification_review":
        return ["1", "2", "3", "4", "5"]
    elif task == "generation_news":
        return []
    elif task == "generation_paper":
        return []
    elif task == "paraphrase_paper":
        return []

def create_preprocessor(tokenizer, max_length):
    def preprocess_dataset(examples):
        inputs = [example for example in examples["source"]]
        targets = [example for example in examples["target"]]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True)
        return model_inputs
    return preprocess_dataset

def convert_to_hf_dataset(dataset, cache_dir):
    def gen():
        for idx in range(len(dataset)):
            yield dataset[idx]
    return datasets.Dataset.from_generator(gen, cache_dir = cache_dir)

def create_input_output_gen_func(task):
    if task == "LaMP-1":
        def func(item):
            inp = f"Write an abstract for this title: {item['title']}"
            out = f'{item["abstract"]}'
            return inp, out
    elif task == "LaMP-2":
        def func(item):
            inp = f"Which tag does this movie relate to among the following tags? Just answer with the tag name without further explanation. tags: [sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story] description: {item['description']}"
            out = f'{item["tag"]}'
            return inp, out
    elif task == "LaMP-3":
        def func(item):
            inp = f"What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: {item['text']}"
            out = f'{item["score"]}'
            return inp, out
    elif task == "LaMP-4":
        def func(item):
            inp = f"Generate a headline for the following article: {item['text']}"
            out = f'{item["title"]}'
            return inp, out
    elif task == "LaMP-5":
        def func(item):
            inp = f"Generate a title for the following abstract of a paper: {item['abstract']}"
            out = f'{item["title"]}'
            return inp, out
    elif task == "LaMP-6":
        def func(item):
            inp = f"Generate a subject for the following email: {item['text']}"
            out = f'{item["title"]}'
            return inp, out
    elif task == "LaMP-7":
        def func(item):
            percent = random.uniform(0.1, 0.25)
            tweet_words = item['text'].split()
            index = int(len(tweet_words) * percent)
            in_inp = " ".join(tweet_words[:index])
            in_out = " ".join(tweet_words[index:])
            inp = f"Complete the following tweet: {in_inp}"
            out = f'{in_out}'
            return inp, out
    return func


def create_per_user_dataset(data_addr, user_ids, task, cache_dir):
    with open(data_addr) as file:
        orig_dataset = json.load(file)
    seen_users = set()
    datasets = dict()
    input_output_gen_func = create_input_output_gen_func(task)
    for data in orig_dataset:
        uid = str(data['user_id'])
        if user_ids is not None and uid not in user_ids:
            continue
        if uid in seen_users:
            continue
        else:
            seen_users.add(uid)
        cur_dataset = []
        for i, item in enumerate(data['profile']):
            id = f'{uid}-{data["id"]}-{i}'
            inp, out = input_output_gen_func(item)
            cur_dataset.append(
                {
                    "id" : id,
                    "input" : inp,
                    "output" : out
                }
            )
        datasets[uid] = convert_to_hf_dataset(GeneralSeq2SeqDataset(cur_dataset), cache_dir)
    return datasets

def create_per_user_dataset_test(data_addr, user_ids, task, cache_dir):
    with open(data_addr) as file:
        orig_dataset = json.load(file)
    seen_users = set()
    datasets = dict()
    for data in orig_dataset:
        uid = str(data['user_id'])
        if user_ids is not None and uid not in user_ids:
            continue
        elif uid not in seen_users:
            seen_users.add(uid)
            datasets[uid] = []
        
        datasets[uid].append(
            {
                "id" : data["id"],
                "input" : data["input"],
                "output" : data["output"]
            }
        )
    
    for key, value in datasets.items():
        datasets[key] = convert_to_hf_dataset(GeneralSeq2SeqDataset(value), cache_dir)
    return datasets

class GeneralSeq2SeqDataset(Dataset):

    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return {
            "id" : self.data[index]['id'],
            "source" : self.data[index]['input'],
            "target" : self.data[index]['output']
        }
    
    def __len__(self):
        return len(self.data)