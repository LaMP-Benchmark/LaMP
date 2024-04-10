# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import types
import torch
from transformers import PreTrainedModel, AutoModel
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
import numpy as np

class RSPG(PreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.model = AutoModel.from_pretrained(config.init_model)
        self.classifier = nn.Linear(self.model.config.hidden_size, 1)
    
    def forward(self, input_ids = None, attention_mask = None, token_type_ids = None, **kwargs):
        output = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
        )
        return self.classifier(output.pooler_output).view(-1, self.config.num_labels)

class Trainer(nn.Module):

    def __init__(self, model, temperature = 1.0):
        super().__init__()
        self.model = model
        self.loss_fn_kl = nn.KLDivLoss(reduction="batchmean")
        self.temperature = temperature
    
    def forward(
            self,
            input_ids = None,
            attention_mask = None,
            token_type_ids = None,
            labels = None
        ):

        outputs = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
        )

        loss = self.loss_fn_kl(nn.functional.log_softmax(outputs, dim = -1) / self.temperature, nn.functional.softmax(labels, dim = -1))
        
        return loss, outputs