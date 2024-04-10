from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch.nn import DataParallel
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
import math


def select_elements(tensor, k):
    B = tensor.size(0)
    selected_rows = []
    for i in range(B):
        start_idx = i * k
        end_idx = (i + 1) * k
        selected_rows.append(tensor[i, start_idx:end_idx])
    selected_tensor = torch.stack(selected_rows)
    return selected_tensor

def loss_fn_reinforce(probs, rewards):
    x = torch.mul(probs, rewards)
    x = torch.sum(x) / (probs.shape[0] * probs.shape[1])
    return -x

class KDReaderToRetrieverTrainer(nn.Module):

    def __init__(self, model, args) -> None:
        super().__init__()
        self.model = model
        self.args = args
        
    def forward(
            self, 
            query_input_ids = None,
            query_token_type_ids = None,
            query_attention_mask = None,
            documents_input_ids = None,
            documents_token_type_ids = None,
            documents_attention_mask = None,
            documents_ctxs = None,
            scores_gold = None,
            target_txt = None,
            **kws
        ):
        return self._forward(
            query_input_ids,
            query_token_type_ids,
            query_attention_mask,
            documents_input_ids,
            documents_token_type_ids,
            documents_attention_mask,
            documents_ctxs,
            target_txt,
            scores_gold
        )
    
    def _forward(
            self, 
            query_input_ids,
            query_token_type_ids,
            query_attention_mask,
            documents_input_ids,
            documents_token_type_ids,
            documents_attention_mask,
            documents_ctxs,
            target_txt,
            scores_gold,
            **kws
        ):
        B = documents_token_type_ids.shape[0]
        ctx_size = documents_token_type_ids.shape[1]

        query_reps = self.model(input_ids = query_input_ids, token_type_ids = query_token_type_ids, attention_mask = query_attention_mask)
        docs_reps = self.model(input_ids = documents_input_ids.view(B * ctx_size, -1), token_type_ids = documents_token_type_ids.view(B * ctx_size, -1), attention_mask = documents_attention_mask.view(B * ctx_size, -1))
        docs_reps = docs_reps.view(B, ctx_size, -1)
        scores = torch.einsum("ij,ikj->ik", query_reps, docs_reps)
        probs = torch.softmax(scores / self.args.temperature, dim = -1)
        
        # gold label creation
        if self.args.greater_is_better:
            gold_scores = torch.zeros_like(scores, device = scores.device)
            for i, sample in enumerate(scores_gold):
                for j, score in enumerate(sample):
                    gold_scores[i, j] = (score[self.args.reader_gold_metric])
        else:
            target_numerical = [int(x) for x in target_txt]
            worst_score_array = [max(abs(x-1), abs(x-5)) for x in target_numerical]
            gold_scores = torch.tensor([[float(x) for y in range(ctx_size)] for x in worst_score_array], device = scores.device)
            for i, sample in enumerate(scores_gold):
                for j, score in enumerate(sample):
                    gold_scores[i, j] = (worst_score_array[i] - score[self.args.reader_gold_metric]) / worst_score_array[i]
        
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(probs, torch.softmax(gold_scores  / self.args.temperature, dim = -1))
        
        
        return loss, torch.softmax(scores, dim = -1), gold_scores 
    
class RLReaderToRetrieverTrainer(nn.Module):

    def __init__(self, model, args) -> None:
        super().__init__()
        self.model = model
        self.args = args
        
    def forward(
            self, 
            query_input_ids = None,
            query_token_type_ids = None,
            query_attention_mask = None,
            documents_input_ids = None,
            documents_token_type_ids = None,
            documents_attention_mask = None,
            documents_ctxs = None,
            scores_gold = None,
            target_txt = None,
            **kws
        ):
        return self._forward(
            query_input_ids,
            query_token_type_ids,
            query_attention_mask,
            documents_input_ids,
            documents_token_type_ids,
            documents_attention_mask,
            documents_ctxs,
            target_txt,
            scores_gold
        )
        
    def _forward(
            self, 
            query_input_ids,
            query_token_type_ids,
            query_attention_mask,
            documents_input_ids,
            documents_token_type_ids,
            documents_attention_mask,
            documents_ctxs,
            target_txt,
            scores_gold,
            **kws
        ):
        B = documents_token_type_ids.shape[0]
        ctx_size = documents_token_type_ids.shape[1]

        query_reps = self.model(input_ids = query_input_ids, token_type_ids = query_token_type_ids, attention_mask = query_attention_mask)
        docs_reps = self.model(input_ids = documents_input_ids.view(B * ctx_size, -1), token_type_ids = documents_token_type_ids.view(B * ctx_size, -1), attention_mask = documents_attention_mask.view(B * ctx_size, -1))
        docs_reps = docs_reps.view(B, ctx_size, -1)
        scores = torch.einsum("ij,ikj->ik", query_reps, docs_reps)
        probs = torch.softmax(scores, dim = -1)
        
        sample_idx = probs.multinomial(1)
        
        # gold label creation
        if self.args.greater_is_better:
            gold_scores = torch.zeros_like(scores, device = scores.device)
            for i, sample in enumerate(scores_gold):
                for j, score in enumerate(sample):
                    gold_scores[i, j] = (score[self.args.reader_gold_metric] - sample[0][self.args.reader_gold_metric])
        else:
            target_numerical = [int(x) for x in target_txt]
            worst_score_array = [max(abs(x-1), abs(x-5)) for x in target_numerical]
            gold_scores = torch.tensor([[float(x) for y in range(ctx_size)] for x in worst_score_array], device = scores.device)
            for i, sample in enumerate(scores_gold):
                for j, score in enumerate(sample):
                    gold_scores[i, j] = (sample[0][self.args.reader_gold_metric] - score[self.args.reader_gold_metric]) / worst_score_array[i]
        
        probs = torch.gather(probs, 1, sample_idx)
        gold_scores = torch.gather(gold_scores, 1, sample_idx)
        loss = loss_fn_reinforce(torch.log(probs), gold_scores)
        
        return loss, torch.softmax(scores, dim = -1), gold_scores 