from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from ..dataset import SequenceBatch

class _Encoder(nn.Module):
    def __init__(self, embedding, output_size):
        super().__init__()
        vocab_size, embedding_size = embedding.shape[0], embedding.shape[1]

        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_size,
                                      padding_idx=0, _weight=torch.Tensor(embedding))
        self.rnn = nn.LSTM(embedding_size, output_size // 2, batch_first=True, bidirectional=True)

    def forward(self, x, length):
        h1 = self.embedding(x)
        h1_packed = nn.utils.rnn.pack_padded_sequence(h1, length.cpu(), batch_first=True, enforce_sorted=False)
        h2_packed, _ = self.rnn(h1_packed)
        h2_unpacked, _ = nn.utils.rnn.pad_packed_sequence(h2_packed, batch_first=True, padding_value=0.0)
        return h1, h2_unpacked

class _Attention(nn.Module):
    def __init__(self,
                 input_size=128,
                 attention_hidden_size=128):
        super().__init__()
        self.W1 = nn.Linear(input_size, attention_hidden_size)
        self.W2 = nn.Linear(attention_hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def score(self, values):
        # x.shape == (batch_size, max_length, input_size)
        # h_t.shape == (batch_size, max_length, attention_hidden_size)
        h_t = torch.tanh(self.W1(values))
        # score_t.shape = (batch_size, max_length, 1)
        score_t = self.W2(h_t)
        return score_t.squeeze(2) # (batch_size, max_length)

    def forward(self, values, mask):
        # values.shape == (batch_size, max_length, hidden_size)
        # score_t.shape = (batch_size, max_length)
        score_t = self.score(values)

        # Compute masked attention weights, given the score values.
        # alpha_t.shape = (batch_size, max_length)
        # Mask = False, indicates that data should be ignored
        score_t.masked_fill_(torch.logical_not(mask), -np.inf)
        alpha_t = self.softmax(score_t)

        # Compute context vector
        # context_t.shape = (batch_size, hidden_size)
        context_t = (alpha_t.unsqueeze(2) * values).sum(1)

        return context_t, alpha_t

class SingleSequenceToClass(pl.LightningModule):
    """Implements the Text-Classification task from 'Attention is not Explanation'

    The paper's model code is in:
        https://github.com/successar/AttentionExplanation/blob/master/model/Binary_Classification.py
    The code is very complex because they integrate all their analysis.

    However, in general:
    * Uses a Single layer Bi-LSTM Encoder
    * Uses a Linear layer as Decoder
    * Uses Additive-tanh-attention as Attention

    Hyper parameters:
    * LSTM-hidden-size is 128 (https://github.com/successar/AttentionExplanation/blob/master/configurations.py#L31)
    * weight_decay is 1e-5 (https://github.com/successar/AttentionExplanation/blob/master/configurations.py#L20)
    * Learning algorithm is `torch.optim.Adam(lr=0.001, weight_decay=1e-5, amsgrad=True)`
        (https://github.com/successar/AttentionExplanation/blob/master/model/Binary_Classification.py#L83)
    * Weight decay is only applied to encoder and decoder (not attention)

    Differences from 'Attention Interpretablity Across NLP Tasks':
    * Uses Glove embeddings
    """
    def __init__(self, embedding, hidden_size=128, num_of_classes=2):
        super().__init__()
        self.encoder = _Encoder(embedding, 2 * hidden_size)
        self.attention = _Attention(2 * hidden_size, hidden_size)
        self.decoder = nn.Linear(2 * hidden_size, num_of_classes)
        self.ce_loss = nn.CrossEntropyLoss()

        self.val_metric_acc = pl.metrics.Accuracy(compute_on_step=False)
        self.val_metric_auroc = pl.metrics.AUROC(num_classes=num_of_classes, compute_on_step=False)
        self.val_metric_f1 = pl.metrics.F1(num_classes=num_of_classes, average='macro', compute_on_step=False)
        self.test_metric_acc = pl.metrics.Accuracy(compute_on_step=False)
        self.test_metric_auroc = pl.metrics.AUROC(num_classes=num_of_classes, compute_on_step=False)
        self.test_metric_f1 = pl.metrics.F1(num_classes=num_of_classes, average='macro', compute_on_step=False)

    @property
    def embedding_matrix(self):
        return self.encoder.embedding.weight.data

    def forward(self, batch: SequenceBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Mask = True, indicates to use. Mask = False, indicates should be ignored.
        embedding, h1 = self.encoder(batch.sentence, batch.length)
        h2, alpha = self.attention(h1, batch.mask)
        h3 = self.decoder(h2)
        return h3, alpha, embedding

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

    def test_step(self, batch: SequenceBatch, batch_nb):
        y, _, _ = self.forward(batch)
        predict_label = torch.argmax(y, dim=1)
        predict_prop = F.softmax(y, dim=1)
        self.test_metric_acc.update(predict_label, batch.label)
        self.test_metric_auroc.update(predict_prop, batch.label)
        self.test_metric_f1.update(predict_label, batch.label)

    def validation_epoch_end(self, outputs):
        self.log(f'acc_val', self.val_metric_acc.compute(), on_epoch=True)
        self.log(f'auroc_val', self.val_metric_auroc.compute(), on_epoch=True, prog_bar=True)
        self.log(f'f1_val', self.val_metric_f1.compute(), on_epoch=True, prog_bar=True)
        self.val_metric_acc.reset()
        self.val_metric_auroc.reset()
        self.val_metric_f1.reset()

    def test_epoch_end(self, outputs):
        self.log(f'acc_test', self.test_metric_acc.compute(), on_epoch=True)
        self.log(f'auroc_test', self.test_metric_auroc.compute(), on_epoch=True, prog_bar=True)
        self.log(f'f1_test', self.test_metric_f1.compute(), on_epoch=True, prog_bar=True)
        self.test_metric_acc.reset()
        self.test_metric_auroc.reset()
        self.test_metric_f1.reset()

    def configure_optimizers(self):
        return torch.optim.Adam([
            {'params': self.encoder.parameters(), 'weight_decay': 1e-5 },
            {'params': self.attention.parameters() },
            {'params': self.decoder.parameters(), 'weight_decay': 1e-5 },
        ], lr=0.001, amsgrad=True)
