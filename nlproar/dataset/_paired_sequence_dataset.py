from collections import namedtuple

import torch

from ._dataset import Dataset, SequenceBatch

class PairedSequenceDataset(Dataset):
    def _pickle_data_to_torch_data(self, data):
        return [{
            'sentence': torch.tensor(self.tokenizer.truncate(x['sentence']), dtype=torch.int64),
            'mask': torch.tensor(self.tokenizer.truncate(self.tokenizer.mask(x['sentence'])), dtype=torch.bool),
            'length': torch.tensor(len(self.tokenizer.truncate(x['sentence'])), dtype=torch.int64),

            'sentence_aux': torch.tensor(self.tokenizer.truncate(x['sentence_aux']), dtype=torch.int64),
            'sentence_aux_mask': torch.tensor(self.tokenizer.truncate(self.tokenizer.mask(x['sentence_aux'])), dtype=torch.bool),
            'sentence_aux_length': torch.tensor(len(self.tokenizer.truncate(x['sentence_aux'])), dtype=torch.int64),

            'sentence_pair': torch.tensor(self.tokenizer.truncate(self.tokenizer.sentence_pair(x['sentence'], x['sentence_aux'])), dtype=torch.int64),
            'sentence_type': torch.tensor(self.tokenizer.truncate(self.tokenizer.sentence_type(x['sentence'], x['sentence_aux'])), dtype=torch.int64),

            'label': torch.tensor(x['label'], dtype=torch.int64),
            'index': torch.tensor(idx, dtype=torch.int64)
        } for idx, x
          in enumerate(data)]

    def collate(self, observations) -> SequenceBatch:
        return SequenceBatch(
            sentence=self.tokenizer.stack_pad([observation['sentence'] for observation in observations]),
            length=torch.stack([observation['length'] for observation in observations]),
            mask=self.tokenizer.stack_pad_mask([observation['mask'] for observation in observations]),

            sentence_aux=self.tokenizer.stack_pad([observation['sentence_aux'] for observation in observations]),
            sentence_aux_length=torch.stack([observation['sentence_aux_length'] for observation in observations]),
            sentence_aux_mask=self.tokenizer.stack_pad_mask([observation['sentence_aux_mask'] for observation in observations]),

            sentence_pair=self.tokenizer.stack_pad([observation['sentence_pair'] for observation in observations]),
            sentence_type=self.tokenizer.stack_pad_mask([observation['sentence_type'] for observation in observations]),

            label=torch.stack([observation['label'] for observation in observations]),
            index=torch.stack([observation['index'] for observation in observations])
        )

    def uncollate(self, batch):
        return [{
            'sentence': sentence[:length],
            'mask': mask[:length],
            'length': length,
            'sentence_aux': sentence_aux[:sentence_aux_length],
            'sentence_aux_mask': sentence_aux_mask[:sentence_aux_length],
            'sentence_aux_length': sentence_aux_length,
            'label': label,
            'index': index
        } for sentence, mask, length,
              sentence_aux, sentence_aux_mask, sentence_aux_length,
              label, index
          in zip(batch.sentence, batch.mask, batch.length,
                 batch.sentence_aux, batch.sentence_aux_mask, batch.sentence_aux_length,
                 batch.label, batch.index)]
