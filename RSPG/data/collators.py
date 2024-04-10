import json
import datasets
import torch

class RSPGPreCollator(object):

    def __init__(self, tokenizer, max_length) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_length
        
    def __call__(self, batch):

        inps = [x for ex in batch for x in ex['inputs']]
        inps = self.tokenizer.batch_encode_plus(
            inps,
            max_length=self.max_len,
            padding=True,
            return_tensors='pt',
            truncation=True,
        )
        labels = torch.tensor([x['labels'] for x in batch])
        

        return {
            "id" : [x['id'] for x in batch],
            "input_ids" : inps['input_ids'],
            "attention_mask" : inps['attention_mask'],
            "labels" : labels,
            "outputs" : [x['outputs'] for x in batch],
            "gold" : [x['gold'] for x in batch]
        }

class RSPGPostCollator(object):

    def __init__(self, tokenizer, max_length) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_length
        
    def __call__(self, batch):

        inps = [f"{x} {self.tokenizer.sep_token} {y}" for ex in batch for x, y in zip(ex['inputs'], ex['outputs'])]
        inps = self.tokenizer.batch_encode_plus(
            inps,
            max_length=self.max_len,
            padding=True,
            return_tensors='pt',
            truncation=True,
        )
        labels = torch.tensor([x['labels'] for x in batch])
        

        return {
            "id" : [x['id'] for x in batch],
            "input_ids" : inps['input_ids'],
            "attention_mask" : inps['attention_mask'],
            "labels" : labels,
            "outputs" : [x['outputs'] for x in batch],
            "gold" : [x['gold'] for x in batch]
        }