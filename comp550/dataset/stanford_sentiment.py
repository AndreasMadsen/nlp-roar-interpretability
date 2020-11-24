
import os.path as path
import re
import os
import random
import pickle
import warnings

import torchtext
import torch
import spacy
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class _Tokenizer:
    def __init__(self):
        self.ids_to_token = []
        self.token_to_ids = {}

        self.pad_token = '[PAD]'
        self.mask_token = '[MASK]'
        self.start_token = '[CLS]'
        self.end_token = '[EOS]'
        self.unknown_token = '[UNK]'
        self.special_symbols = [
            self.pad_token, self.mask_token,
            self.start_token, self.end_token,
            self.unknown_token
        ]

        self._tokenizer = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])

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
            tokens |= set(self.tokenizer(sentence))

        self.ids_to_token = self.special_symbols + list(tokens)
        self._update_token_to_ids()

    def tokenizer(self, sentence):
        sentence = sentence.strip()
        sentence = sentence.replace("-LRB-", '')
        sentence = sentence.replace("-RRB-", '  ')
        sentence = re.sub(r'\W+', ' ', sentence)
        return [t.text.lower() for t in self._tokenizer(sentence)]

    def encode(self, sentence):
        return [2] + [
            self.token_to_ids.get(word, 4)
            for word in self.tokenizer(sentence)
        ] + [3]

    def decode(self, token_ids):
        return ' '.join([
            self.ids_to_token.get(token_id)
            for token_id in token_ids
        ])

    def stack_pad(self, observations):
        max_length = max(tokens.shape[0] for tokens in observations)
        padded_observations = [
            torch.cat([tokens, torch.zeros(max_length - tokens.shape[0], dtype=tokens.dtype)], dim=0)
            for tokens in observations
        ]
        return torch.stack(padded_observations)

class StanfordSentimentDataset(pl.LightningDataModule):
    """Loads the Stanford Sentiment Dataset

    Uses the same tokenization procedure as in "Attention is not Explanation"

    The paper's tokenizer can be found in:
        https://github.com/successar/AttentionExplanation/blob/master/preprocess/SST/SST.ipynb
    In general:
     * Uses spacy tokenizer
     * Lower case tokens
     * Does not drop tokens
     * Replaces \W = [^a-zA-Z0-9_] with <space>
     * Removes "-LRB-"
     * Replaces "-RRB-" with <space>
     * Removes sentences shorter than 5

    The paper's embedding code is in:
        https://github.com/successar/AttentionExplanation/blob/master/preprocess/vectorizer.py#L103
    In general:
    * use 'fasttext.simple.300d'
    * set [PAD] embedding to zero
    * if word does not exist in 'fasttext.simple.300d' use `np.random.randn(300)`

    Batch-size is defined in:
    * https://github.com/successar/AttentionExplanation/blob/master/configurations.py#L19

    Min-length is defined in:
    * https://github.com/successar/AttentionExplanation/blob/master/Trainers/DatasetBC.py#L103
    """
    def __init__(self, cachedir, batch_size=32, seed=0, num_workers=4):
        super().__init__()
        self._cachedir = path.realpath(cachedir)
        self._batch_size = batch_size
        self._seed = seed
        self._num_workers = 4
        self.tokenizer = _Tokenizer()
        self.label_names = ['negative', 'positive']
        self._setup_complete = False

    @property
    def vocabulary(self):
        return self.tokenizer.ids_to_token

    def embedding(self):
        lookup = torchtext.vocab.pretrained_aliases['fasttext.simple.300d'](cache=self._cachedir + '/embeddings')
        rng = np.random.RandomState(self._seed)

        embeddings = []
        for word in self.tokenizer.ids_to_token:
            if word == self.tokenizer.pad_token:
                embeddings.append(np.zeros(300))
            elif word in set(self.tokenizer.special_symbols) or word not in lookup.stoi:
                embeddings.append(rng.randn(300))
            else:
                embeddings.append(lookup[word].numpy())

        return np.vstack(embeddings)

    def prepare_data(self):
        # Load embeddings
        torchtext.vocab.pretrained_aliases['fasttext.simple.300d'](cache=self._cachedir + '/embeddings')

        # Load dataset
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            train, val, test = torchtext.datasets.SST.splits(
                torchtext.data.Field(), torchtext.data.Field(sequential=False),
                filter_pred=lambda ex: len(ex.text) > 5 and ex.label != 'neutral',
                root=self._cachedir + '/datasets')

        # Create vocabulary from training data, if it hasn't already been done
        if not path.exists(self._cachedir + '/vocab/sst.vocab'):
            os.makedirs(self._cachedir + '/vocab', exist_ok=True)

            self.tokenizer.from_iterable(' '.join(row.text) for row in train)
            self.tokenizer.to_file(self._cachedir + '/vocab/sst.vocab')
        else:
            self.tokenizer.from_file(self._cachedir + '/vocab/sst.vocab')

        # Encode data
        if not path.exists(self._cachedir + '/encoded/sst.pkl'):
            os.makedirs(self._cachedir + '/encoded', exist_ok=True)

            rng = random.Random(self._seed)
            data = {}
            for name, dataset in [('train', train), ('val', val), ('test', test)]:
                observations = []
                for index, observation in enumerate(dataset):
                    observations.append({
                        'sentence': self.tokenizer.encode(' '.join(observation.text)),
                        'label': self.label_names.index(observation.label),
                        'index': index
                    })
                data[name] = rng.sample(observations, len(observations))

            with open(self._cachedir + '/encoded/sst.pkl', 'wb') as fp:
                pickle.dump(data, fp)

    def _process_data(self, data):
        return [{
            'sentence': torch.tensor(x['sentence'], dtype=torch.int64),
            'label': torch.tensor(x['label'], dtype=torch.int64),
            'index': torch.tensor(x['index'], dtype=torch.int64)
        } for x in data]

    def setup(self, stage=None):
        if not self._setup_complete:
            with open(self._cachedir + '/encoded/sst.pkl', 'rb') as fp:
                data = pickle.load(fp)
            self._train = self._process_data(data['train'])
            self._val = self._process_data(data['val'])
            self._test = self._process_data(data['test'])
            self._setup_complete = True

    def _collate(self, observations):
        return {
            'sentence': self.tokenizer.stack_pad([observation['sentence'] for observation in observations]),
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
