import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


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

    def forward(self, x, length):

        h1 = self.embedding(x)
        h1_packed = nn.utils.rnn.pack_padded_sequence(
            h1, length, batch_first=True, enforce_sorted=False)
        h2_packed, (h, c) = self.rnn(h1_packed)
        last_hidden = torch.cat([h[0], h[1]], dim=-1)
        h2_unpacked, _ = nn.utils.rnn.pad_packed_sequence(
            h2_packed, batch_first=True, padding_value=0)
        return h2_unpacked, last_hidden


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
        # x.shape == (batch_size, max_length, input_size)
        # h_t.shape == (batch_size, max_length, attention_hidden_size)
        h_t = torch.tanh(self.W1_premise(premise_hidden) +
                         self.W1_hypothesis(hypothesis_hidden).unsqueeze(1))
        # score_t.shape = (batch_size, max_length, 1)
        score_t = self.W2(h_t)
        return score_t.squeeze(2)  # (batch_size, max_length)

    def forward(self, premise_hidden, hypothesis_hidden, premise_mask):
        # values.shape == (batch_size, max_length, hidden_size)
        # score_t.shape = (batch_size, max_length)
        score_t = self.score(premise_hidden, hypothesis_hidden)

        # Compute masked attention weights, given the score values.
        # alpha_t.shape = (batch_size, max_length)
        score_t.masked_fill_(premise_mask, -np.inf)
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

    def forward(self, batch):
        h1_premise, _ = self.encoder_premise(
            batch['premise'], batch['premise_length'])
        _, last_hidden_hypothesis = self.encoder_hypothesis(
            batch['hypothesis'], batch['hypothesis_length'])
        h2, alpha = self.attention(
            h1_premise, last_hidden_hypothesis, batch['premise_mask'])
        predict = self.decoder(h2, last_hidden_hypothesis)
        return predict, alpha

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
            F.one_hot(target, predict.shape[1]).cpu().numpy(),
            F.softmax(predict, dim=1).cpu().numpy()), dtype=torch.float32)
        self.log(f'auc_{name}', auc, on_epoch=True, prog_bar=True)

        f1 = torch.tensor(sklearn.metrics.f1_score(
            target.cpu().numpy(),
            predict_label.cpu().numpy(),
            average='macro'), dtype=torch.float32)
        self.log(f'f1_{name}', f1, on_epoch=True, prog_bar=True)

    def validation_epoch_end(self, outputs):
        return self._valid_test_epoch_end(outputs, name='val')

    def test_epoch_end(self, outputs):
        return self._valid_test_epoch_end(outputs, name='test')

    def configure_optimizers(self):
        '''
        Weight decay is applied on all parameters for SNLI
        https://github.com/successar/AttentionExplanation/blob/425a89a49a8b3bffc3f5e8338287e2ecd0cf1fa2/model/Question_Answering.py#L98
        '''
        return torch.optim.Adam(list(self.parameters()), weight_decay=1e-5, amsgrad=True)
