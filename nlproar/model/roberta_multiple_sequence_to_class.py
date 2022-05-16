from typing import Tuple, Optional
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import transformers

from ._base_multiple_sequence_to_class import BaseMultipleSequenceToClass
from ..dataset import SequenceBatch

class RobertaMultipleSequenceToClass(BaseMultipleSequenceToClass):

    def __init__(self, cachedir, embedding, hidden_size=None, num_of_classes=3):
        """Creates a model instance that maps from a sequence pair to a class

        Args:
            embedding (np.array): The inital word embedding matrix, for example Glove
            hidden_size (int, optional): The hidden size used in the attention mechanism. Defaults to 128.
            num_of_classes (int, optional): The number of output classes. Defaults to 3.
        """
        super().__init__(num_of_classes=num_of_classes)

        self.config = transformers.RobertaConfig.from_pretrained(
            "roberta-base",
            cache_dir=f'{cachedir}/huggingface/transformers')
        self.config.num_labels = num_of_classes
        self.model = transformers.RobertaForSequenceClassification.from_pretrained(
            'roberta-base',
            cache_dir=f'{cachedir}/huggingface/transformers',
            config=self.config)
        self.embedding = self.model.roberta.embeddings.word_embeddings

    @property
    def embedding_matrix(self):
        return self.embedding.weight.data

    def flatten_parameters(self):
        pass

    def forward(self, batch: SequenceBatch, embedding_scale: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = self.embedding(batch.sentence_pair)
        embedding.requires_grad_()

        # Scale input embeddings
        if embedding_scale is None:
            inputs_embeds = embedding
        else:
            embedding_scale_type = torch.where(
                batch.sentence_type >= 1,
                torch.ones_like(batch.sentence_type, dtype=torch.float32),
                torch.full_like(batch.sentence_type, embedding_scale, dtype=torch.float32))
            inputs_embeds = embedding * torch.unsqueeze(embedding_scale_type, -1)

        # RobertaTokenizerFast.from_pretrained("roberta-base").pad_token_id == 1
        attention_mask = (batch.sentence_pair != self.config.pad_token_id).type(batch.sentence.dtype)

        predict = self.model.forward(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds
        ).logits

        return predict, None, embedding

    def configure_optimizers(self):
        '''
        Weight decay is applied on all parameters for SNLI and bAbI
        https://github.com/successar/AttentionExplanation/blob/425a89a49a8b3bffc3f5e8338287e2ecd0cf1fa2/model/Question_Answering.py#L98
        '''

        # Create optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=2e-5, betas=(0.9, 0.98)
        )

        # Create scheduler
        train_dataloader = self.trainer.datamodule.train_dataloader()
        batches = len(train_dataloader)
        effective_accum = self.trainer.accumulate_grad_batches
        num_training_steps = (batches // effective_accum) * self.trainer.max_epochs

        scheduler = {
            "scheduler": transformers.get_linear_schedule_with_warmup(
                optimizer,
                int(num_training_steps * 0.06),
                num_training_steps
            ),
            "interval": "step",
            "frequency": 1
        }

        return [optimizer], [scheduler]
