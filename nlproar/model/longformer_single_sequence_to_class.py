from typing import Tuple, Optional

import torch
import transformers

from ._base_single_sequence_to_class import BaseSingleSequenceToClass
from ..dataset import SequenceBatch

class LongformerSingleSequenceToClass(BaseSingleSequenceToClass):

    def __init__(self, cachedir, embedding, hidden_size=None, num_of_classes=2):
        """Creates a model instance that maps from a sequence pair to a class

        Args:
            num_of_classes (int, optional): The number of output classes. Defaults to 3.
        """
        super().__init__(num_of_classes=num_of_classes)

        self.config = transformers.LongformerConfig.from_pretrained(
            "allenai/longformer-base-4096",
            cache_dir=f'{cachedir}/huggingface/transformers')
        self.config.num_labels = num_of_classes
        self.model = transformers.LongformerForSequenceClassification.from_pretrained(
            'allenai/longformer-base-4096',
            cache_dir=f'{cachedir}/huggingface/transformers',
            config=self.config)
        self.embedding = self.model.longformer.embeddings.word_embeddings

    @property
    def embedding_matrix(self):
        return self.embedding.weight.data

    def flatten_parameters(self):
        pass

    def forward(self, batch: SequenceBatch, embedding_scale: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = self.embedding(batch.sentence)
        embedding.requires_grad_()

        # Scale input embeddings
        if embedding_scale is None:
            inputs_embeds = embedding
        else:
            inputs_embeds = embedding * embedding_scale

        # Mask the padding tokens, note that we don't use batch.sentence_mask because this
        # is for ROAR which also masks <s>, </s>, [MASK].
        attention_mask = (batch.sentence != self.config.pad_token_id).type(batch.sentence.dtype)
        global_attention_mask = (batch.sentence == self.config.bos_token_id).type(batch.sentence.dtype)

        predict = self.model.forward(
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
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
