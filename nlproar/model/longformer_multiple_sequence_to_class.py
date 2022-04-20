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

class LongformerMultipleSequenceToClass(BaseMultipleSequenceToClass):

    def __init__(self, cachedir, hidden_size=None, num_of_classes=3):
        """Creates a model instance that maps from a sequence pair to a class

        Args:
            embedding (np.array): The inital word embedding matrix, for example Glove
            hidden_size (int, optional): The hidden size used in the attention mechanism. Defaults to 128.
            num_of_classes (int, optional): The number of output classes. Defaults to 3.
        """
        super().__init__(num_of_classes=num_of_classes)

        self.config = transformers.RobertaConfig.from_pretrained(
            "allenai/longformer-base-4096",
            cache_dir=f'{cachedir}/huggingface/transformers')
        self.config.num_labels = num_of_classes
        self.model = transformers.RobertaForSequenceClassification.from_pretrained(
            "allenai/longformer-base-4096",
            cache_dir=f'{cachedir}/huggingface/transformers',
            config=self.config)
        self.embedding = self.model.roberta.embeddings.word_embeddings

    @property
    def embedding_matrix(self):
        return self.embedding.weight.data

    def flatten_parameters(self):
        pass

    def forward(self, batch: SequenceBatch, embedding_scale: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO:

        embedding = self.embedding(batch.sentence_pair)
        embedding.requires_grad_()

        # Scale input embeddings
        if embedding_scale is None:
            inputs_embeds = embedding
        else:
            inputs_embeds = torch.where(batch.sentence_pair_type, embedding, embedding * embedding_scale)

        # RobertaTokenizerFast.from_pretrained("roberta-base").pad_token_id == 1
        attention_mask = (batch.sentence_pair != self.config.pad_token_id).type(batch.sentence.dtype)
        global_attention_mask = (batch.sentence_pair == self.config.cls_token_id).type(batch.sentence.dtype)

        predict = self.model.forward(
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            inputs_embeds=inputs_embeds
        ).logits

        # The ROAR procedure only needs the embeddings from the first sentence, and expect
        # the shape to match batch.sentence. Note, that the ROAR procedure will also mask
        # with batch.sentence_mask which will mask the second sentence.
        embedding_main = embedding[:, 0:batch.sentence.size(1)]
        return predict, None, embedding_main

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
