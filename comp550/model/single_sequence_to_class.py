
import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class _Encoder(nn.Module):
    def __init__(self, embedding, output_size):
        super().__init__()
        vocab_size, embedding_size = embedding.shape[0], embedding.shape[1]

        self.embedding = nn.Embedding(vocab_size, embedding_size,
                                      padding_idx=0, _weight=torch.Tensor(embedding))
        self.rnn = nn.LSTM(embedding_size, output_size // 2, batch_first=True, bidirectional=True)

    def forward(self, x, length):
        h1 = self.embedding(x)
        h1_packed = nn.utils.rnn.pack_padded_sequence(h1, length, batch_first=True, enforce_sorted=False)
        h2_packed, _ = self.rnn(h1_packed)
        h2_unpacked, _ = nn.utils.rnn.pad_packed_sequence(h2_packed, batch_first=True, padding_value=0)
        return h2_unpacked

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

    def forward(self, batch):
        # Mask = True, indicates to use. Mask = False, indicates should be ignored.
        h1 = self.encoder(batch['sentence'], batch['length'])
        h2, alpha = self.attention(h1, batch['mask'])
        h3 = self.decoder(h2)
        return h3, alpha

    def training_step(self, batch, batch_idx):
        y, alpha = self.forward(batch)
        train_loss = self.ce_loss(y, batch['label'])
        self.log('loss_train', train_loss, on_step=True)
        return train_loss

    def _valid_test_step(self, batch):
        y, alpha = self.forward(batch)
        return {
            'predict': y,
            'target': batch['label']
        }

    def validation_step(self, batch, batch_nb):
        return self._valid_test_step(batch)

    def test_step(self, batch, batch_nb):
        return self._valid_test_step(batch)

    def _valid_test_epoch_end(self, outputs, name='val'):
        predict = torch.cat([output['predict'] for output in outputs], dim=0)
        target = torch.cat([output['target'] for output in outputs], dim=0)
        predict_label = torch.argmax(predict, dim=1)

        loss = self.ce_loss(predict, target)
        self.log(f'loss_{name}', loss, on_epoch=True, prog_bar=True)

        acc = torch.mean((predict_label == target).type(torch.float32))
        self.log(f'acc_{name}', acc, on_epoch=True)

        auc = torch.tensor(sklearn.metrics.roc_auc_score(
            F.one_hot(target, predict.shape[1]).numpy(),
            F.softmax(predict, dim=1).numpy()), dtype=torch.float32)
        self.log(f'auc_{name}', auc, on_epoch=True, prog_bar=True)

        f1 = torch.tensor(sklearn.metrics.f1_score(
            target.numpy(),
            predict_label.numpy(),
            average='macro'), dtype=torch.float32)
        self.log(f'f1_{name}', f1, on_epoch=True, prog_bar=True)

    def validation_epoch_end(self, outputs):
        return self._valid_test_epoch_end(outputs, name='val')

    def test_epoch_end(self, outputs):
        return self._valid_test_epoch_end(outputs, name='test')

    def configure_optimizers(self):
        return torch.optim.Adam([
            {'params': self.encoder.parameters(), 'weight_decay': 1e-5 },
            {'params': self.attention.parameters() },
            {'params': self.decoder.parameters(), 'weight_decay': 1e-5 },
        ], lr=0.001, amsgrad=True)
