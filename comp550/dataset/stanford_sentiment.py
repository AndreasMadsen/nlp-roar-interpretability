
import os.path as path
import re
import os

import torchtext
import torch
import spacy
import numpy as np
import pytorch_lightning as pl
from datasets import load_dataset
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
            self.ids_to_token = list(line.strip() for line in fp)
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
        self._cachedir = cachedir
        self._batch_size = batch_size
        self._seed = seed
        self._num_workers = 4
        self.tokenizer = _Tokenizer()
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
        datasets = load_dataset('glue', 'sst2', cache_dir=self._cachedir + '/datasets')

        # Create vocabulary from training data, if it hasn't already been done
        if not path.exists(self._cachedir + '/vocab/sst2.vocab'):
            os.makedirs(self._cachedir + '/vocab', exist_ok=True)

            self.tokenizer.from_iterable(row['sentence'] for row in datasets['train'])
            self.tokenizer.to_file(self._cachedir + '/vocab/sst2.vocab')
        else:
            self.tokenizer.from_file(self._cachedir + '/vocab/sst2.vocab')

    def _process_dataset(self, dataset):
        dataset = dataset.map(lambda x: {
            'sentence': self.tokenizer.encode(x['sentence']),
            'label': x['label']
        })
        dataset = dataset.filter(lambda x: len(x['sentence']) >= 5)
        dataset = dataset.shuffle(seed=self._seed)
        dataset.set_format(type='torch', columns=['sentence', 'label'])
        return dataset

    def setup(self, stage=None):
        if not self._setup_complete:
            datsets = load_dataset('glue', 'sst2', cache_dir=self._cachedir + '/datasets')
            self._train = self._process_dataset(datsets['train'])
            self._validation = self._process_dataset(datsets['validation'])
            self._test = self._process_dataset(datsets['test'])
            self._setup_complete = True

    def _collate(self, observations):
        return {
            'sentence': self.tokenizer.stack_pad([observation['sentence'] for observation in observations]),
            'label': torch.stack([observation['label'] for observation in observations])
        }

    def train_dataloader(self):
        return DataLoader(self._train, batch_size=self._batch_size, collate_fn=self._collate, num_workers=self._num_workers)

    def val_dataloader(self):
        return DataLoader(self._validation, batch_size=self._batch_size, collate_fn=self._collate, num_workers=self._num_workers)

    def test_dataloader(self):
        return DataLoader(self._test, batch_size=self._batch_size, collate_fn=self._collate, num_workers=self._num_workers)
