from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from ..dataset import SequenceBatch

class _Decoder(nn.Module):
    def __init__(self, hidden_size=128, num_of_classes=3):
        super().__init__()
        self.decoder_premise = nn.Linear(2 * hidden_size, hidden_size)
        self.decoder_hypothesis = nn.Linear(2 * hidden_size, hidden_size)
        self.decoder_final = nn.Linear(hidden_size, num_of_classes)

    def forward(self, context, hypothesis_hidden):
        predict = self.decoder_final(torch.tanh(self.decoder_premise(
            context) + self.decoder_hypothesis(hypothesis_hidden)))

        return predict


class _Encoder(nn.Module):
    def __init__(self, embedding, hidden_size):
        super().__init__()
        vocab_size, embedding_size = embedding.shape[0], embedding.shape[1]

        self.embedding = nn.Embedding(vocab_size, embedding_size,
                                      padding_idx=0, _weight=torch.Tensor(embedding))
        self.rnn = nn.LSTM(embedding_size, hidden_size,
                           batch_first=True, bidirectional=True)

    def forward(self, x, length, embedding_scale: Optional[float]):
        h1 = self.embedding(x)
        if embedding_scale is not None:
            h1 = h1 * embedding_scale
        h1_packed = nn.utils.rnn.pack_padded_sequence(
            h1, length.cpu(), batch_first=True, enforce_sorted=False)

        h2_packed, (h, c) = self.rnn(h1_packed)
        last_hidden = torch.cat([h[0], h[1]], dim=-1)
        h2_unpacked, _ = nn.utils.rnn.pad_packed_sequence(
            h2_packed, batch_first=True, padding_value=0.0)

        return h2_unpacked, last_hidden, h1


class _Attention(nn.Module):
    def __init__(self,
                 input_size=128,
                 attention_hidden_size=128):
        super().__init__()
        self.W1_premise = nn.Linear(input_size, attention_hidden_size)
        self.W1_hypothesis = nn.Linear(input_size, attention_hidden_size)
        self.W2 = nn.Linear(
            attention_hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def score(self, premise_hidden, hypothesis_hidden):
        # premise_hidden.shape == (batch_size, max_length, input_size)
        # hypothesis_hidden.shape == (batch_size, input_size)
        # h_t.shape == (batch_size, max_length, attention_hidden_size)
        h_t = torch.tanh(self.W1_premise(premise_hidden) +
                         self.W1_hypothesis(hypothesis_hidden).unsqueeze(1))
        # score_t.shape = (batch_size, max_length, 1)
        score_t = self.W2(h_t)
        return score_t.squeeze(2)  # (batch_size, max_length)

    def forward(self, premise_hidden, hypothesis_hidden, premise_mask):
        # premise_hidden.shape == (batch_size, max_length, hidden_size)
        # hypothesis_hidden.shape == (batch_size, hidden_size)
        # score_t.shape = (batch_size, max_length)
        score_t = self.score(premise_hidden, hypothesis_hidden)

        # Compute masked attention weights, given the score values.
        # alpha_t.shape = (batch_size, max_length)
        score_t.masked_fill_(torch.logical_not(premise_mask), -np.inf)
        alpha_t = self.softmax(score_t)

        # Compute context vector
        # context_t.shape = (batch_size, hidden_size)
        context_t = (alpha_t.unsqueeze(2) * premise_hidden).sum(1)

        return context_t, alpha_t


class MultipleSequenceToClass(pl.LightningModule):

    def __init__(self, embedding, hidden_size=128, num_of_classes=3):
        super().__init__()
        self.encoder_premise = _Encoder(embedding, hidden_size)
        self.encoder_hypothesis = _Encoder(embedding, hidden_size)
        self.attention = _Attention(
            2 * hidden_size, hidden_size)
        self.decoder = _Decoder(hidden_size, num_of_classes)
        self.ce_loss = nn.CrossEntropyLoss()

        self.val_metric_acc = pl.metrics.Accuracy(compute_on_step=False)
        self.val_metric_f1 = pl.metrics.F1(num_classes=num_of_classes, average='micro', compute_on_step=False)
        self.test_metric_acc = pl.metrics.Accuracy(compute_on_step=False)
        self.test_metric_f1 = pl.metrics.F1(num_classes=num_of_classes, average='micro', compute_on_step=False)

    @property
    def embedding_matrix(self):
        return self.encoder_premise.embedding.weight.data

    def forward(self, batch: SequenceBatch, embedding_scale: Optional[float]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h1_premise, _, embedding = self.encoder_premise(
            batch.sentence, batch.length, embedding_scale)
        _, last_hidden_hypothesis, _ = self.encoder_hypothesis(
            batch.sentence_aux, batch.sentence_aux_length)
        h2, alpha = self.attention(
            h1_premise, last_hidden_hypothesis, batch.mask)
        predict = self.decoder(h2, last_hidden_hypothesis)
        return predict, alpha, embedding

    def training_step(self, batch: SequenceBatch, batch_idx):
        y, _, _ = self.forward(batch)
        train_loss = self.ce_loss(y, batch.label)
        self.log('loss_train', train_loss, on_step=True)
        return train_loss

    def validation_step(self, batch: SequenceBatch, batch_nb):
        predict = torch.argmax(self.forward(batch)[0], dim=1)
        self.val_metric_acc.update(predict, batch.label)
        self.val_metric_f1.update(predict, batch.label)

    def test_step(self, batch: SequenceBatch, batch_nb):
        predict = torch.argmax(self.forward(batch)[0], dim=1)
        self.test_metric_acc.update(predict, batch.label)
        self.test_metric_f1.update(predict, batch.label)

    def validation_epoch_end(self, outputs):
        self.log(f'acc_val', self.val_metric_acc.compute(), on_epoch=True)
        self.log(f'f1_val', self.val_metric_f1.compute(), on_epoch=True, prog_bar=True)
        self.val_metric_acc.reset()
        self.val_metric_f1.reset()

    def test_epoch_end(self, outputs):
        self.log(f'acc_test', self.test_metric_acc.compute(), on_epoch=True)
        self.log(f'f1_test', self.test_metric_f1.compute(), on_epoch=True, prog_bar=True)
        self.test_metric_acc.reset()
        self.test_metric_f1.reset()

    def configure_optimizers(self):
        '''
        Weight decay is applied on all parameters for SNLI and bAbI
        https://github.com/successar/AttentionExplanation/blob/425a89a49a8b3bffc3f5e8338287e2ecd0cf1fa2/model/Question_Answering.py#L98
        '''
        return torch.optim.Adam(self.parameters(), weight_decay=1e-5, amsgrad=True)
