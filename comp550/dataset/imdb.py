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
import requests

from torch.utils.data import Dataset
from .tokenizer import Tokenizer


class IMDBTokenizer(Tokenizer):
    def __init__(self):
        '''
        Document frequency is 10
        https://github.com/successar/AttentionExplanation/blob/master/preprocess/IMDB/IMDB.ipynb
        '''
        super().__init__()
        self.min_df = 10

    def tokenize(self, sentence):
        return sentence.split()


class IMDBDataModule(pl.LightningDataModule):

    def __init__(self, cachedir, batch_size=32, num_workers=4):
        super().__init__()
        self._cachedir = cachedir
        self._batch_size = batch_size
        self._num_workers = num_workers
        self.tokenizer = IMDBTokenizer()

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

        if not path.exists(self._cachedir + '/text-datasets/imdb_full.pkl'):
            os.makedirs(self._cachedir + '/text-datasets', exist_ok=True)
            r = requests.get(imdb_data_s3_url)
            with open(self._cachedir + '/text-datasets/imdb_full.pkl', 'wb') as f:
                f.write(r.content)

        if not path.exists(self._cachedir + '/text-datasets/imdb_word_index.json'):
            r = requests.get(imdb_vocab_s3_url)
            with open(self._cachedir + '/text-datasets/imdb_word_index.json', 'wb') as f:
                f.write(r.content)

        if not path.exists(self._cachedir + '/text-datasets/imdb_full_text.pkl'):
            
            imdb_data = pickle.load(open(self._cachedir + '/text-datasets/imdb_full.pkl', 'rb'))
            word_index = json.load(open(self._cachedir + '/text-datasets/imdb_word_index.json'))

            train_set, test_set = imdb_data

            idx_to_word_map = {idx:word for word, idx in word_index.items()}

            '''
            In the original implementation:
             * Sentences with length greater than or equal to 400 are removed
             * Only 20% of test set is used
            https://github.com/successar/AttentionExplanation/blob/master/preprocess/IMDB/IMDB.ipynb 
            '''
            trainidx = [i for i, x in enumerate(train_set[0]) if len(x) < 400]
            trainidx, devidx = train_test_split(trainidx, train_size=0.8)
            testidx = [i for i, x in enumerate(test_set[0]) if len(x) < 400]
            testidx, _ = train_test_split(testidx, train_size=0.2)

            imdb_data = {}
            for name, idxs, dataset in [('train', trainidx, train_set), ('val', devidx, train_set), ('test', testidx, test_set)]:
                '''
                Min length is 6 
                https://github.com/successar/AttentionExplanation/blob/425a89a49a8b3bffc3f5e8338287e2ecd0cf1fa2/Trainers/DatasetBC.py#L108
                '''
                imdb_data[name] = [{
                    'sentence': " ".join([idx_to_word_map[x] for x in dataset[0][idx]]),
                    'label': dataset[1][idx]
                } for idx in idxs if len(dataset[0][idx]) > 6]

            with open(self._cachedir + '/text-datasets/imdb_full_text.pkl', 'wb') as fp:
                pickle.dump(imdb_data, fp)

        torchtext.vocab.pretrained_aliases['fasttext.simple.300d'](
            cache=self._cachedir + '/embeddings')

        if not path.exists(self._cachedir + '/vocab/imdb.vocab'):
            os.makedirs(self._cachedir + '/vocab', exist_ok=True)

            imdb_data = pickle.load(open(self._cachedir + '/text-datasets/imdb_full_text.pkl', 'rb'))
            self.tokenizer.from_iterable(x['sentence'] for x in imdb_data['train'])

            self.tokenizer.to_file(self._cachedir + '/vocab/imdb.vocab')
        else:
            self.tokenizer.from_file(self._cachedir + '/vocab/imdb.vocab')

        if not path.exists(self._cachedir + '/encoded/imdb.pkl'):
            os.makedirs(self._cachedir + '/encoded', exist_ok=True)
            
            imdb_data = pickle.load(open(self._cachedir + '/text-datasets/imdb_full_text.pkl', 'rb'))

            data = {}
            for name in ['train', 'val', 'test']:

                dataset = imdb_data[name]
                data[name] = [{
                    'sentence': self.tokenizer.encode(instance['sentence']),
                    'label': instance['label']
                } for instance in dataset]

            with open(self._cachedir + '/encoded/imdb.pkl', 'wb') as fp:
                pickle.dump(data, fp)

    def setup(self, stage=None):
        with open(self._cachedir + '/encoded/imdb.pkl', 'rb') as fp:
            dataset = pickle.load(fp)
        if stage == "fit":
            self._train = self._process_data(dataset['train'])
            self._val = self._process_data(dataset['val'])
        elif stage == 'test':
            self._test = self._process_data(dataset['test'])
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

    def collate(self, observations):
        return {
            'sentence': self.tokenizer.stack_pad([observation['sentence'] for observation in observations]),
            'length': [observation['length'] for observation in observations],
            'mask': self.tokenizer.stack_pad([observation['mask'] for observation in observations]),
            'label': torch.stack([observation['label'] for observation in observations]),
            'index': torch.stack([observation['index'] for observation in observations])
        }

    def uncollate(self, batch):
        return [{
            'sentence': sentence[:length],
            'mask': mask[:length],
            'length': length,
            'label': label,
            'index': index
        } for sentence, mask, length,
              label, index
          in zip(batch['sentence'], batch['mask'], batch['length'],
                 batch['label'], batch['index'])]

    def train_dataloader(self):
        return DataLoader(self._train,
                          batch_size=self._batch_size, collate_fn=self.collate,
                          num_workers=self._num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self._val,
                          batch_size=self._batch_size, collate_fn=self.collate,
                          num_workers=self._num_workers)

    def test_dataloader(self):
        return DataLoader(self._test,
                          batch_size=self._batch_size, collate_fn=self.collate,
                          num_workers=self._num_workers)
