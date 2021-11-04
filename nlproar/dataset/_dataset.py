from collections import namedtuple

import pickle
import os.path as path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from ._tokenizer import Tokenizer

class SequenceBatch(namedtuple('SequenceBatch', [
    'sentence', 'length', 'mask', 'sentence_aux',
    'sentence_aux_length', 'sentence_aux_mask', 'label', 'index'
])):
    def cuda(self):
        return self._make(val.cuda() for val in self)

    def pin_memory(self):
        return self._make(val.pin_memory() for val in self)

class Dataset(pl.LightningDataModule):
    def __init__(self, cachedir, name, tokenizer, seed=0, batch_size=32, num_workers=4):
        super().__init__()
        self._cachedir = path.realpath(cachedir)
        self.name = name
        self.batch_size = batch_size

        self._seed = seed
        self._np_rng = np.random.RandomState(seed)
        self._num_workers = num_workers

        self.tokenizer = tokenizer

    @property
    def vocabulary(self):
        return self.tokenizer.ids_to_token

    def embedding(self):
        raise NotImplementedError('embedding method is missing')

    def prepare_data(self):
        raise NotImplementedError('prepare_data method is missing')

    def _pickle_data_to_torch_data(self):
        raise NotImplementedError('_pickle_data_to_torch_data method is missing')

    def setup(self, stage=None):
        with open(self._cachedir + f'/encoded/{self.name}.pkl', 'rb') as fp:
            data = pickle.load(fp)
        if stage == 'fit':
            self._train = self._pickle_data_to_torch_data(data['train'])
            self._val = self._pickle_data_to_torch_data(data['val'])
        elif stage == 'test':
            self._test = self._pickle_data_to_torch_data(data['test'])
        else:
            raise ValueError(f'unexpected setup stage: {stage}')

    def is_setup(self, stage):
        if stage == 'fit':
            return hasattr(self, '_train') and hasattr(self, '_val')
        elif stage == 'test':
            return hasattr(self, '_test')
        else:
            raise ValueError(f'unexpected setup stage: {stage}')

    def clean(self, stage=None):
        if stage == 'fit':
            del self._train
            del self._val
        elif stage == 'test':
            del self._test
        else:
            raise ValueError(f'unexpected setup stage: {stage}')

    def train_dataloader(self, batch_size=None, num_workers=None, shuffle=True):
        return DataLoader(self._train,
                          batch_size=batch_size or self.batch_size, collate_fn=self.collate,
                          num_workers=self._num_workers if num_workers is None else num_workers,
                          shuffle=shuffle, pin_memory=True)

    def val_dataloader(self, batch_size=None, num_workers=None, shuffle=False):
        return DataLoader(self._val,
                          batch_size=batch_size or self.batch_size, collate_fn=self.collate,
                          num_workers=self._num_workers if num_workers is None else num_workers,
                          shuffle=shuffle, pin_memory=True)

    def test_dataloader(self, batch_size=None, num_workers=None, shuffle=False):
        return DataLoader(self._test,
                          batch_size=batch_size or self.batch_size, collate_fn=self.collate,
                          num_workers=self._num_workers if num_workers is None else num_workers,
                          shuffle=shuffle, pin_memory=True)

    def dataloader(self, split, *args, **kwargs):
        return getattr(self, f'{split}_dataloader')(*args, **kwargs)

    def num_of_observations(self, split):
        return len(getattr(self, f'_{split}'))
