from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl

from ._total_ce_loss import TotalCrossEntropyLoss
from ..dataset import SequenceBatch

class BaseMultipleSequenceToClass(pl.LightningModule):

    def __init__(self, num_of_classes=3):
        """Creates a model instance that maps from a sequence pair to a class

        Args:
            num_of_classes (int, optional): The number of output classes. Defaults to 3.
        """
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()

        self.val_metric_acc = torchmetrics.Accuracy(compute_on_step=False)
        self.val_metric_f1 = torchmetrics.F1(num_classes=num_of_classes, average='micro', compute_on_step=False)
        self.val_metric_ce = TotalCrossEntropyLoss()

        self.test_metric_acc = torchmetrics.Accuracy(compute_on_step=False)
        self.test_metric_f1 = torchmetrics.F1(num_classes=num_of_classes, average='micro', compute_on_step=False)
        self.test_metric_ce = TotalCrossEntropyLoss()

    @property
    def embedding_matrix(self):
        raise NotImplementedError

    def flatten_parameters(self):
        raise NotImplementedError

    def training_step(self, batch: SequenceBatch, batch_idx):
        y, _, _ = self.forward(batch)
        train_loss = self.ce_loss(y, batch.label)
        self.log('loss_train', train_loss, on_step=True)
        return train_loss

    def validation_step(self, batch: SequenceBatch, batch_nb):
        y, _, _ = self.forward(batch)
        predict = torch.argmax(y, dim=1)
        self.val_metric_acc.update(predict, batch.label)
        self.val_metric_f1.update(predict, batch.label)
        self.val_metric_ce.update(y, batch.label)

    def test_step(self, batch: SequenceBatch, batch_nb):
        y, _, _ = self.forward(batch)
        predict = torch.argmax(y, dim=1)
        self.test_metric_acc.update(predict, batch.label)
        self.test_metric_f1.update(predict, batch.label)
        self.test_metric_ce.update(y, batch.label)

    def validation_epoch_end(self, outputs):
        self.log('acc_val', self.val_metric_acc.compute(), on_epoch=True)
        self.log('f1_val', self.val_metric_f1.compute(), on_epoch=True, prog_bar=True)
        self.log('ce_val', self.val_metric_ce.compute(), on_epoch=True)
        self.val_metric_acc.reset()
        self.val_metric_f1.reset()
        self.val_metric_ce.reset()

    def test_epoch_end(self, outputs):
        self.log('acc_test', self.test_metric_acc.compute(), on_epoch=True)
        self.log('f1_test', self.test_metric_f1.compute(), on_epoch=True, prog_bar=True)
        self.log('ce_test', self.test_metric_ce.compute(), on_epoch=True)
        self.test_metric_acc.reset()
        self.test_metric_f1.reset()
        self.test_metric_ce.reset()
