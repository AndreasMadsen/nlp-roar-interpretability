from typing import Tuple, Optional
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import transformers

from ._total_ce_loss import TotalCrossEntropyLoss
from ..dataset import SequenceBatch

class RobertaSingleSequenceToClass(pl.LightningModule):

    def __init__(self, cachedir, hidden_size=128, num_of_classes=2):
        """Creates a model instance that maps from a sequence pair to a class

        Args:
            embedding (np.array): The inital word embedding matrix, for example Glove
            hidden_size (int, optional): The hidden size used in the attention mechanism. Defaults to 128.
            num_of_classes (int, optional): The number of output classes. Defaults to 3.
        """
        super().__init__()

        self.config = transformers.RobertaConfig.from_pretrained(
            "roberta-base",
            cache_dir=f'{cachedir}/huggingface/transformers')
        self.config.num_labels = num_of_classes
        self.model = transformers.RobertaForSequenceClassification.from_pretrained(
            'roberta-base',
            cache_dir=f'{cachedir}/huggingface/transformers',
            config=self.config)
        self.embedding = self.model.roberta.embeddings.word_embeddings
        self.ce_loss = nn.CrossEntropyLoss()

        self.val_metric_acc = torchmetrics.Accuracy(compute_on_step=False)
        self.val_metric_f1 = torchmetrics.F1(num_classes=num_of_classes, average='macro', compute_on_step=False)
        self.val_metric_ce = TotalCrossEntropyLoss()

        self.test_metric_acc = torchmetrics.Accuracy(compute_on_step=False)
        self.test_metric_f1 = torchmetrics.F1(num_classes=num_of_classes, average='macro', compute_on_step=False)
        self.test_metric_ce = TotalCrossEntropyLoss()

        with warnings.catch_warnings():
            # Ignore this warning:
            # Metric `AUROC` will save all targets and predictions in buffer.
            # For large datasets this may lead to large memory footprint.
            warnings.filterwarnings("ignore", category=UserWarning)
            self.val_metric_auroc = torchmetrics.AUROC(num_classes=num_of_classes, compute_on_step=False)
            self.test_metric_auroc = torchmetrics.AUROC(num_classes=num_of_classes, compute_on_step=False)

    @property
    def embedding_matrix(self):
        return self.embedding.weight.data

    def flatten_parameters(self):
        pass

    def forward(self, batch: SequenceBatch, embedding_scale: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = self.embedding(batch.sentence)
        embedding.requires_grad_()

        # Mask the padding tokens, note that we don't use batch.sentence_mask because this
        # is for ROAR which also masks <s>, </s>, [MASK].
        attention_mask = (batch.sentence != self.config.pad_token_id).type(batch.sentence.dtype)

        predict = self.model.forward(
            attention_mask=attention_mask,
            inputs_embeds=embedding if embedding_scale is None else embedding * embedding_scale
        ).logits

        return predict, None, embedding

    def training_step(self, batch: SequenceBatch, batch_idx):
        y, _, _ = self.forward(batch)
        train_loss = self.ce_loss(y, batch.label)
        self.log('loss_train', train_loss, on_step=True)
        return train_loss

    def validation_step(self, batch: SequenceBatch, batch_nb):
        y, _, _ = self.forward(batch)
        predict_label = torch.argmax(y, dim=1)
        predict_prop = F.softmax(y, dim=1)
        self.val_metric_acc.update(predict_label, batch.label)
        self.val_metric_auroc.update(predict_prop, batch.label)
        self.val_metric_f1.update(predict_label, batch.label)
        self.val_metric_ce.update(y, batch.label)

    def test_step(self, batch: SequenceBatch, batch_nb):
        y, _, _ = self.forward(batch)
        predict_label = torch.argmax(y, dim=1)
        predict_prop = F.softmax(y, dim=1)
        self.test_metric_acc.update(predict_label, batch.label)
        self.test_metric_auroc.update(predict_prop, batch.label)
        self.test_metric_f1.update(predict_label, batch.label)
        self.test_metric_ce.update(y, batch.label)

    def validation_epoch_end(self, outputs):
        self.log('acc_val', self.val_metric_acc.compute(), on_epoch=True)
        self.log('auroc_val', self.val_metric_auroc.compute(), on_epoch=True, prog_bar=True)
        self.log('f1_val', self.val_metric_f1.compute(), on_epoch=True, prog_bar=True)
        self.log('ce_val', self.val_metric_ce.compute(), on_epoch=True)
        self.val_metric_acc.reset()
        self.val_metric_auroc.reset()
        self.val_metric_f1.reset()
        self.val_metric_ce.reset()

    def test_epoch_end(self, outputs):
        self.log('acc_test', self.test_metric_acc.compute(), on_epoch=True)
        self.log('auroc_test', self.test_metric_auroc.compute(), on_epoch=True, prog_bar=True)
        self.log('f1_test', self.test_metric_f1.compute(), on_epoch=True, prog_bar=True)
        self.log('ce_test', self.test_metric_ce.compute(), on_epoch=True)
        self.test_metric_acc.reset()
        self.test_metric_auroc.reset()
        self.test_metric_f1.reset()
        self.test_metric_ce.reset()

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
