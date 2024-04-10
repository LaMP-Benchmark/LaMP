from torch.utils.data import Dataset
import json
import datasets
import numpy as np
import random

class RSPGDataset(Dataset):

    def __init__(self, data_addr, smaller_is_better = False) -> None:
        super().__init__()
        with open(data_addr) as file:
            data = json.load(file)
        self.data = data
        self.smaller_is_better = smaller_is_better

    def __getitem__(self, index):
        data = self.data[index]
        did = data['id']
        if self.smaller_is_better:
            worst_score = max(abs(int(data['gold'])-1), abs(int(data['gold'])-5))
            labels = [(worst_score - x) / worst_score for x in self.data[index]['labels']]
        else:
            labels = self.data[index]['labels']
        outputs = self.data[index]['outputs']

        return {
            "id" : did,
            "inputs" : [x.lower() for x in data['inputs']],
            "labels" : labels,
            "outputs" : outputs,
            "gold" : data['gold']
        }
    
    def __len__(self):
        return len(self.data)