from collections import namedtuple

import torch

from ._dataset import Dataset, SequenceBatch

empty_tensor = torch.tensor([])

class SingleSequenceDataset(Dataset):
    def _pickle_data_to_torch_data(self, data):
        return [{
            'sentence': torch.tensor(x['sentence'], dtype=torch.int64),
            'length': torch.tensor(len(x['sentence']), dtype=torch.int64),
            'mask': torch.tensor(self.tokenizer.mask(x['sentence']), dtype=torch.bool),
            'label': torch.tensor(x['label'], dtype=torch.int64),
            'index': torch.tensor(idx, dtype=torch.int64)
        } for idx, x in enumerate(data)]

    def collate(self, observations) -> SequenceBatch:
        return SequenceBatch(
            sentence=self.tokenizer.stack_pad([observation['sentence'] for observation in observations]),
            length=torch.stack([observation['length'] for observation in observations]),
            mask=self.tokenizer.stack_pad([observation['mask'] for observation in observations]),
            sentence_aux=empty_tensor,
            sentence_aux_length=empty_tensor,
            sentence_aux_mask=empty_tensor,
            label=torch.stack([observation['label'] for observation in observations]),
            index=torch.stack([observation['index'] for observation in observations])
        )

    def uncollate(self, batch):
        return [{
            'sentence': sentence[:length],
            'mask': mask[:length],
            'length': length,
            'label': label,
            'index': index
        } for sentence, mask, length,
              label, index
          in zip(batch.sentence, batch.mask, batch.length,
                 batch.label, batch.index)]
