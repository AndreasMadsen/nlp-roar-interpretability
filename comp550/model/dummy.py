
import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class DummyModel(pl.LightningModule):
    """Throws if called
    """
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        raise RuntimeError('DummyModel should not be called')

