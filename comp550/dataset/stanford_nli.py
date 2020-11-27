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

from datasets import load_dataset
from torch.utils.data import Dataset


class _Tokenizer:
    def __init__(self):
        '''
        Original implementation has "en", we are using "en_core_web_sm"
        https://github.com/successar/AttentionExplanation/blob/master/preprocess/vectorizer.py
        https://github.com/successar/AttentionExplanation/blob/master/preprocess/SST/SST.ipynb
        '''
        self.ids_to_token = []
        self.token_to_ids = {}

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
        self.special_symbols = [
            self.pad_token,
            self.start_token, self.end_token,
            self.mask_token, self.unknown_token
        ]

        self._tokenizer = spacy.load('en_core_web_sm', disable=[
                                     'parser', 'tagger', 'ner'])

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
        tokens = set()
        for sentence in iterable:
            tokens |= set(self.tokenize(sentence))

        self.ids_to_token = self.special_symbols + list(tokens)
        self._update_token_to_ids()

    def tokenize(self, sentence):

        sentence = re.sub(r"\s+", " ", sentence.strip())
        sentence = [t.text.lower() for t in self._tokenizer(sentence)]
        '''
        TODO Why is this needed?
        https://github.com/successar/AttentionExplanation/blob/425a89a49a8b3bffc3f5e8338287e2ecd0cf1fa2/preprocess/vectorizer.py#L23
        '''
        sentence = ["qqq" if any(char.isdigit()
                                 for char in word) else word for word in sentence]
        return sentence

    def encode(self, sentence):
        return [self.start_token_id] + [
            self.token_to_ids.get(word, 4)
            for word in self.tokenize(sentence)
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


class SNLIDataModule(pl.LightningDataModule):

    '''
    batch size is 128
    https://github.com/successar/AttentionExplanation/blob/425a89a49a8b3bffc3f5e8338287e2ecd0cf1fa2/Trainers/DatasetQA.py#L94
    '''

    def __init__(self, cachedir, batch_size=128, num_workers=4):
        super().__init__()
        self._cachedir = cachedir
        self._batch_size = batch_size
        self._num_workers = num_workers
        self.tokenizer = _Tokenizer()

    @property
    def vocabulary(self):
        return self.tokenizer.ids_to_token

    def embedding(self):
        lookup = torchtext.vocab.pretrained_aliases['glove.840B.300d'](
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
        In the Hugging Face distribution of the dataset,
        the label has 4 possible values, 0, 1, 2, -1.
        which correspond to entailment, neutral, contradiction,
        and no label respectively.

        https://github.com/huggingface/datasets/tree/master/datasets/snli
        '''
        snli_dataset = load_dataset(
            'snli', cache_dir=self._cachedir + '/datasets')
        torchtext.vocab.pretrained_aliases['glove.840B.300d'](
            cache=self._cachedir + '/embeddings')

        # Create vocabulary from training data, if it hasn't already been done
        if not path.exists(self._cachedir + '/vocab/snli.vocab'):
            os.makedirs(self._cachedir + '/vocab', exist_ok=True)

            self.tokenizer.from_iterable(
                " ".join([row['premise'], row["hypothesis"]]) for row in snli_dataset['train'])
            self.tokenizer.to_file(self._cachedir + '/vocab/snli.vocab')
        else:
            self.tokenizer.from_file(self._cachedir + '/vocab/snli.vocab')

        if not path.exists(self._cachedir + '/encoded/snli.pkl'):
            os.makedirs(self._cachedir + '/encoded', exist_ok=True)

            data = {}
            for name, dataset in [('train', snli_dataset['train']), ('val', snli_dataset['validation']), ('test', snli_dataset['test'])]:
                '''
                Change in data statistics on removing "no label"
                Train - 550,152 -> 549367
                Valid - 10,000 -> 9842
                Test - 10,000 -> 9824
                '''
                dataset = dataset.filter(lambda x: x['label'] != -1)
                data[name] = list(dataset.map(lambda x: {
                    'premise': self.tokenizer.encode(x['premise']),
                    'hypothesis': self.tokenizer.encode(x['hypothesis']),
                    'label': x['label'],
                }))

            with open(self._cachedir + '/encoded/snli.pkl', 'wb') as fp:
                pickle.dump(data, fp)

    def setup(self, stage=None):
        with open(self._cachedir + '/encoded/snli.pkl', 'rb') as fp:
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
            'premise': torch.tensor(x['premise'], dtype=torch.int64),
            'premise_length': len(x['premise']),
            'premise_mask': torch.tensor(self.tokenizer.mask(x['premise']), dtype=torch.bool),
            'hypothesis': torch.tensor(x['hypothesis'], dtype=torch.int64),
            'hypothesis_length': len(x['hypothesis']),
            'hypothesis_mask': torch.tensor(self.tokenizer.mask(x['hypothesis']), dtype=torch.bool),
            'label': torch.tensor(x['label'], dtype=torch.int64),
            'index': torch.tensor(idx, dtype=torch.int64)
        } for idx, x in enumerate(data)]

    def _collate(self, observations):
        return {
            'premise': self.tokenizer.stack_pad([observation['premise'] for observation in observations]),
            'premise_length': [observation['premise_length'] for observation in observations],
            'premise_mask': self.tokenizer.stack_pad([observation['premise_mask'] for observation in observations]),
            'hypothesis': self.tokenizer.stack_pad([observation['hypothesis'] for observation in observations]),
            'hypothesis_length': [observation['hypothesis_length'] for observation in observations],
            'hypothesis_mask': self.tokenizer.stack_pad([observation['hypothesis_mask'] for observation in observations]),
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
