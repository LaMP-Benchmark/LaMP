from torch.utils.data import Dataset
import json
import datasets

def get_all_labels(task):
    if task == "LaMP-1":
        return ["[1]","[2]"]
    elif task == "LaMP-2":
        return ['sci-fi', 'based on a book', 'comedy', 'action', 'twist ending', 'dystopia', 'dark comedy', 'classic', 'psychology', 'fantasy', 'romance', 'thought-provoking', 'social commentary', 'violence', 'true story']
    elif task == "LaMP-3":
        return ["1", "2", "3", "4", "5"]
    else:
        return []

def create_preprocessor(tokenizer, max_length):
    def preprocess_dataset(examples):
        inputs = [example for example in examples["source"]]
        targets = [example for example in examples["target"]]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True)
        return model_inputs
    return preprocess_dataset

def create_preprocessor_chatgpt(tokenizer, max_length):
    def preprocess_dataset(examples):
        inputs = [example for example in examples["source"]]
        targets = [example for example in examples["target"]]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True)
        model_inputs = tokenizer.batch_decode(model_inputs['input_ids'], skip_special_tokens=True)
        return {"chatgpt_inputs" : model_inputs}
    return preprocess_dataset

def convert_to_hf_dataset(dataset):
    def gen():
        for idx in range(len(dataset)):
            yield dataset[idx]
    return datasets.Dataset.from_generator(gen)

class GeneralSeq2SeqDataset(Dataset):

    def __init__(self, data_addr, use_profile, task, create_prompt = None) -> None:
        super().__init__()
        with open(data_addr) as file:
            self.data = json.load(file)
        self.use_profile = use_profile
        self.task = task
        assert not (use_profile ^ (create_prompt != None)), "You should provide a prompt maker function when you use profile"
        self.create_prompt = create_prompt

    def __getitem__(self, index):
        if self.use_profile:
            return {
                "source" : self.create_prompt(self.data[index]['input'], self.data[index]['profile'], self.task),
                "target" : self.data[index]['output']
            }
        else:
            return {
                "source" : self.data[index]['input'],
                "target" : self.data[index]['output']
            }
    
    def __len__(self):
        return len(self.data)

class ReaderToRetrieverDataset(Dataset):

    def __init__(self, data_addr, task, create_query_corpus, is_llama = False) -> None:
        super().__init__()
        with open(data_addr) as file:
            self.data = json.load(file)
        self.task = task
        self.create_query_corpus = create_query_corpus
        self.is_llama = is_llama

    def __getitem__(self, index):
        query, corpus = self.create_query_corpus(self.data[index]['input'], self.data[index]['profile'])
        return {
            "qid" : self.data[index]['id'],
            "query_raw" : self.data[index]['input'] + " answer:" if self.is_llama else "",
            "query" : query,
            "documents" : corpus,
            "profile" : self.data[index]['profile'],
            "target" : self.data[index]['output']
        }
    
    def __len__(self):
        return len(self.data)