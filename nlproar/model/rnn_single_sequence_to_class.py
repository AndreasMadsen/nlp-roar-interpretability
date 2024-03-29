from typing import Tuple, Optional
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

from ._total_ce_loss import TotalCrossEntropyLoss
from ..dataset import SequenceBatch
from ._base_single_sequence_to_class import BaseSingleSequenceToClass

class _Encoder(nn.Module):
    def __init__(self, embedding, output_size):
        super().__init__()
        vocab_size, embedding_size = embedding.shape[0], embedding.shape[1]

        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_size,
                                      padding_idx=0, _weight=torch.Tensor(embedding))
        self.rnn = nn.LSTM(embedding_size, output_size // 2, batch_first=True, bidirectional=True)

    def forward(self, x, length, embedding_scale: Optional[torch.Tensor]=None):
        h1 = self.embedding(x)
        h1.requires_grad_()
        h1_scaled = h1 if embedding_scale is None else h1 * embedding_scale
        h1_packed = nn.utils.rnn.pack_padded_sequence(h1_scaled, length.cpu(), batch_first=True, enforce_sorted=False)
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
        score_t = torch.where(mask, score_t, torch.tensor(-np.inf, dtype=score_t.dtype, device=score_t.device))
        alpha_t = self.softmax(score_t)

        # Compute context vector
        # context_t.shape = (batch_size, hidden_size)
        context_t = (alpha_t.unsqueeze(2) * values).sum(1)

        return context_t, alpha_t

class RNNSingleSequenceToClass(BaseSingleSequenceToClass):
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

    def __init__(self, cachedir, embedding, hidden_size=128, num_of_classes=2):
        """Creates a model instance that maps from a single sequence to a class

        Args:
            embedding (np.array): The inital word embedding matrix, for example Glove
            hidden_size (int, optional): The hidden size used in the attention mechanism. Defaults to 128.
            num_of_classes (int, optional): The number of output classes. Defaults to 2.
        """

        super().__init__(num_of_classes=num_of_classes)
        self.encoder = _Encoder(embedding, 2 * hidden_size)
        self.attention = _Attention(2 * hidden_size, hidden_size)
        self.decoder = nn.Linear(2 * hidden_size, num_of_classes)

    @property
    def embedding_matrix(self):
        return self.encoder.embedding.weight.data

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()

    def forward(self, batch: SequenceBatch, embedding_scale: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Mask = True, indicates to use. Mask = False, indicates should be ignored.
        embedding, h1 = self.encoder(batch.sentence, batch.length, embedding_scale)
        h2, alpha = self.attention(h1, batch.mask)
        h3 = self.decoder(h2)
        return h3, alpha, embedding

    def configure_optimizers(self):
        return torch.optim.Adam([
            {'params': self.encoder.parameters(), 'weight_decay': 1e-5 },
            {'params': self.attention.parameters() },
            {'params': self.decoder.parameters(), 'weight_decay': 1e-5 },
        ], lr=0.001, amsgrad=True)
