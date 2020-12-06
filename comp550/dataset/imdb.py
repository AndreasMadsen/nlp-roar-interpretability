import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
import spacy
import re
import torchtext
import os.path as path
import os
import torch
import numpy as np
import pickle
from itertools import chain
from collections import Counter
import json
import subprocess
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset


class _Tokenizer:
    def __init__(self):
        '''
        df is 10
        '''
        self.ids_to_token = []
        self.token_to_ids = {}
        self.min_df = 10

        self.pad_token = '[PAD]'
        self.pad_token_id = 0
        self.start_token = '[CLS]'
        self.start_token_id = 1
        self.end_token = '[EOS]'
        self.end_token_id = 2
        self.mask_token = '[MASK]'
        self.mask_token_id = 3
        self.unknown_token = '[UNK]'
        self.unknown_token_id = 4
        self.digits_token = '[DIGITS]'
        self.digits_token_id = 5
        self.special_symbols = [
            self.pad_token,
            self.start_token, self.end_token,
            self.mask_token, self.unknown_token
        ]

    def _update_token_to_ids(self):
        self.token_to_ids = {
            token: ids for ids, token in enumerate(self.ids_to_token)
        }

    def from_file(self, filepath):
        with open(filepath, 'r') as fp:
            self.ids_to_token = [line.strip() for line in fp]
        self._update_token_to_ids()

    def to_file(self, filepath):
        with open(filepath, 'w') as fp:
            for token in self.ids_to_token:
                print(token, file=fp)

    def from_iterable(self, iterable):
        counter = Counter()
        for sentence in iterable:
            counter.update(set(sentence))

        tokens = [x for x in counter.keys() if counter[x] >= self.min_df]
        self.ids_to_token = self.special_symbols + tokens
        self._update_token_to_ids()

    def encode(self, sentence):
        return [self.start_token_id] + [
            self.token_to_ids.get(word, self.unknown_token_id)
            for word in sentence
        ] + [self.end_token_id]

    def mask(self, token_ids):
        return [token_id >= self.mask_token_id for token_id in token_ids]

    def decode(self, token_ids):
        return ' '.join([
            self.ids_to_token.get(token_id)
            for token_id in token_ids
        ])

    def stack_pad(self, observations):
        max_length = max(tokens.shape[0] for tokens in observations)
        padded_observations = [
            torch.cat([tokens, torch.zeros(
                max_length - tokens.shape[0], dtype=tokens.dtype)], dim=0)
            for tokens in observations
        ]
        return torch.stack(padded_observations)


class IMDBDataModule(pl.LightningDataModule):

    def __init__(self, cachedir, batch_size=32, num_workers=4):
        super().__init__()
        self._cachedir = cachedir
        self._batch_size = batch_size
        self._num_workers = num_workers
        self.tokenizer = _Tokenizer()

    @property
    def vocabulary(self):
        return self.tokenizer.ids_to_token

    def embedding(self):
        lookup = torchtext.vocab.pretrained_aliases['fasttext.simple.300d'](
            cache=self._cachedir + '/embeddings')

        embeddings = []
        for word in self.tokenizer.ids_to_token:
            if word in set(self.tokenizer.special_symbols) or word not in lookup.stoi:
                embeddings.append(np.zeros(300))
            else:
                embeddings.append(lookup[word].numpy())

        return np.vstack(embeddings)

    def prepare_data(self):
        '''
        '''
        imdb_data_s3_url = 'https://s3.amazonaws.com/text-datasets/imdb_full.pkl'
        imdb_vocab_s3_url = 'https://s3.amazonaws.com/text-datasets/imdb_word_index.json'

        subprocess.run(["wget", "-nc", "-P",
                        self._cachedir + '/text-datasets', imdb_data_s3_url])
        subprocess.run(["wget", "-nc", "-P",
                        self._cachedir + '/text-datasets', imdb_vocab_s3_url])

        torchtext.vocab.pretrained_aliases['fasttext.simple.300d'](
            cache=self._cachedir + '/embeddings')

        if not path.exists(self._cachedir + '/vocab/imdb.vocab'):
            os.makedirs(self._cachedir + '/vocab', exist_ok=True)

            imdb_data = pickle.load(
                open(self._cachedir + '/text-datasets/imdb_full.pkl', 'rb'))
            initial_vocab = json.load(
                open(self._cachedir + '/text-datasets/imdb_word_index.json'))
            initial_vocab_inv = {idx: word for word,
                                 idx in initial_vocab.items()}
            train_set, test_set = imdb_data

            trainidx = [i for i, x in enumerate(train_set[0]) if len(x) < 400]
            trainidx, devidx = train_test_split(trainidx, train_size=0.8)

            testidx = [i for i, x in enumerate(test_set[0]) if len(x) < 400]
            testidx, _ = train_test_split(
                testidx, train_size=0.2)

            train_sentences = [[initial_vocab_inv[x]
                                for x in train_set[0][i]] for i in trainidx]

            self.tokenizer.from_iterable(
                sentence for sentence in train_sentences)
            self.tokenizer.to_file(self._cachedir + '/vocab/imdb.vocab')
        else:
            self.tokenizer.from_file(self._cachedir + '/vocab/imdb.vocab')

        if not path.exists(self._cachedir + '/encoded/imdb.pkl'):
            os.makedirs(self._cachedir + '/encoded', exist_ok=True)

            data = {}
            for name, idxs, dataset in [('train', trainidx, train_set), ('val', devidx, train_set), ('test', testidx, test_set)]:
                '''
                Min length is 6 
                https://github.com/successar/AttentionExplanation/blob/425a89a49a8b3bffc3f5e8338287e2ecd0cf1fa2/Trainers/DatasetBC.py#L108
                '''
                data[name] = [{
                    'sentence': self.tokenizer.encode([initial_vocab_inv[x] for x in dataset[0][idx]]),
                    'label': dataset[1][idx]
                } for idx in idxs if len(dataset[0][idx]) > 6]

            with open(self._cachedir + '/encoded/imdb.pkl', 'wb') as fp:
                pickle.dump(data, fp)

    def setup(self, stage=None):
        with open(self._cachedir + '/encoded/imdb.pkl', 'rb') as fp:
            snli_dataset = pickle.load(fp)
        if stage == "fit":
            self._train = self._process_data(snli_dataset['train'])
            self._val = self._process_data(snli_dataset['val'])
        elif stage == 'test':
            self._test = self._process_data(snli_dataset['test'])
        else:
            raise ValueError(f'unexpected setup stage: {stage}')

    def _process_data(self, data):
        return [{
            'sentence': torch.tensor(x['sentence'], dtype=torch.int64),
            'length': len(x['sentence']),
            'mask': torch.tensor(self.tokenizer.mask(x['sentence']), dtype=torch.bool),
            'label': torch.tensor(x['label'], dtype=torch.int64),
            'index': torch.tensor(idx, dtype=torch.int64)
        } for idx, x in enumerate(data)]

    def _collate(self, observations):
        return {
            'sentence': self.tokenizer.stack_pad([observation['sentence'] for observation in observations]),
            'length': [observation['length'] for observation in observations],
            'mask': self.tokenizer.stack_pad([observation['mask'] for observation in observations]),
            'label': torch.stack([observation['label'] for observation in observations]),
            'index': torch.stack([observation['index'] for observation in observations])
        }

    def train_dataloader(self):
        return DataLoader(self._train,
                          batch_size=self._batch_size, collate_fn=self._collate,
                          num_workers=self._num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self._val,
                          batch_size=self._batch_size, collate_fn=self._collate,
                          num_workers=self._num_workers)

    def test_dataloader(self):
        return DataLoader(self._test,
                          batch_size=self._batch_size, collate_fn=self._collate,
                          num_workers=self._num_workers)
