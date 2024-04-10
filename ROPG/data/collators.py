from typing import Any, List
import numpy as np
import torch
import json


class ReaderToRetreieverCollator:

    def __init__(self, tokenizer, query_max_lenght, document_max_length, number_of_ctx, scores_addr = "") -> None:
        self.tokenizer = tokenizer
        self.query_max_lenght = query_max_lenght
        self.document_max_length = document_max_length
        self.number_of_ctx = number_of_ctx
        self.scores_addr = scores_addr
        if scores_addr:
            with open(scores_addr) as file:
                self.scores = json.load(file)
    
    def __call__(self, examples: List[dict]):
        query_tokens = self.tokenizer([x['query'] for x in examples], max_length = self.query_max_lenght, padding = True, return_tensors = 'pt', truncation = True)
        docs = []
        for x in examples:
            temp_docs = []
            for y in x['documents'][:self.number_of_ctx]:
                temp_docs.append(y)
            while len(temp_docs) < self.number_of_ctx:
                temp_docs.append("")
            docs.append(temp_docs)
        
        profiles = []
        for x in examples:
            temp_docs = []
            for y in x['profile'][:self.number_of_ctx]:
                temp_docs.append(y)
            profiles.append(temp_docs)
        
        documents_tokens = self.tokenizer([y for x in docs for y in x], max_length = self.document_max_length, padding = True, return_tensors = 'pt', truncation = True)
        documents_tokens['input_ids'] = documents_tokens['input_ids'].view(len(examples), self.number_of_ctx, -1)
        documents_tokens['token_type_ids'] = documents_tokens['token_type_ids'].view(len(examples), self.number_of_ctx, -1)
        documents_tokens['attention_mask'] = documents_tokens['attention_mask'].view(len(examples), self.number_of_ctx, -1)
        
        scores_batch = []
        for x in examples:
            scores_sample = []
            for prof in x['profile'][:self.number_of_ctx]:
                score = self.scores[f'{x["qid"]}-{prof["id"]}']
                scores_sample.append(score)
            scores_batch.append(scores_sample)
        # print(scores_batch)
        target_txt = [x['target'] for x in examples]       
        ctxs = torch.tensor([[len(x['documents'][:self.number_of_ctx])] for x in examples])
        return {
            "query_input_ids" : query_tokens['input_ids'],
            "query_token_type_ids" : query_tokens['token_type_ids'],
            "query_attention_mask" : query_tokens['attention_mask'],
            "documents_input_ids" : documents_tokens['input_ids'],
            "documents_token_type_ids" : documents_tokens['token_type_ids'],
            "documents_attention_mask" : documents_tokens['attention_mask'],
            "documents_ctxs" : ctxs,
            "batch_docs_text" : profiles,
            "batch_questions_text" : [x['query_raw'] for x in examples],
            "target_txt" : target_txt,
            "scores_gold" : scores_batch
        }
