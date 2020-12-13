
import pickle
import os.path as path
import os
import re
import json
from itertools import chain
from collections import Counter
import requests
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import spacy
import torch
import torchtext
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from .tokenizer import Tokenizer


class SNLITokenizer(Tokenizer):
    def __init__(self):
        '''
        Original implementation has "en", we are using "en_core_web_sm"
        https://github.com/successar/AttentionExplanation/blob/master/preprocess/vectorizer.py
        https://github.com/successar/AttentionExplanation/blob/master/preprocess/SNLI/SNLI.ipynb
        '''
        super().__init__()
        self.min_df = 3

    def tokenize(self, sentence):
        sentence = re.sub(r"\s+", " ", sentence.strip())
        sentence = [t.text.lower() for t in self._tokenizer(sentence)]
        sentence = [self.digits_token if any(char.isdigit()
                                             for char in word) else word for word in sentence]
        return sentence


class SNLIDataModule(pl.LightningDataModule):

    """Loads the Stanford Natural Language Inference Dataset
    Uses the same tokenization procedure as in "Attention is not Explanation"
    The paper's tokenizer can be found in:
        https://github.com/successar/AttentionExplanation/blob/425a89a49a8b3bffc3f5e8338287e2ecd0cf1fa2/preprocess/vectorizer.py#L17
    In general:
     * Uses spacy tokenizer
     * Lower case tokens
     * Drops tokens with document frequency less than 3
     * Replaces all digits with a special token
     * Batch size of 128 (https://github.com/successar/AttentionExplanation/blob/425a89a49a8b3bffc3f5e8338287e2ecd0cf1fa2/Trainers/DatasetQA.py#L94)
    The paper's embedding code is in:
        https://github.com/successar/AttentionExplanation/blob/master/preprocess/vectorizer.py#L119
    In general:
    * use 'glove.840B.300d' (https://github.com/successar/AttentionExplanation/blob/master/preprocess/SNLI/SNLI.ipynb)
    * set [PAD] embedding to zero
    """

    def __init__(self, cachedir, batch_size=128, num_workers=4):
        super().__init__()
        self._cachedir = cachedir
        self._batch_size = batch_size
        self._num_workers = num_workers
        self.label_names = ['entailment', 'contradiction', 'neutral']
        self.tokenizer = SNLITokenizer()

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

        torchtext.vocab.pretrained_aliases['glove.840B.300d'](
            cache=self._cachedir + '/embeddings')

        # Download and parse data
        dataset = {}
        if (not path.exists(self._cachedir + '/vocab/snli.vocab') or
            not path.exists(self._cachedir + '/encoded/snli.pkl')):

            # Download and open zipfile
            data_url = 'https://nlp.stanford.edu/projects/snli/snli_1.0.zip'
            zf = ZipFile(BytesIO(requests.get(data_url).content))

            # Change in data statistics on removing "no label"
            # Train - 550,152 -> 549367
            # Valid - 10,000 -> 9842
            # Test - 10,000 -> 9824
            for zipfile_name in ['train', 'dev', 'test']:
                dataset[zipfile_name] = []

                for line in zf.open(f'snli_1.0/snli_1.0_{zipfile_name}.jsonl'):
                    observation = json.loads(line)
                    if observation['gold_label'] == '-':
                        continue

                    dataset[zipfile_name].append({
                        'premise': observation['sentence1'],
                        'hypothesis': observation['sentence2'],
                        'label': observation['gold_label']
                    })

        # Create vocabulary from training data, if it hasn't already been done
        if not path.exists(self._cachedir + '/vocab/snli.vocab'):
            os.makedirs(self._cachedir + '/vocab', exist_ok=True)

            self.tokenizer.from_iterable(chain.from_iterable(
                (row['premise'], row["hypothesis"]) for row in dataset['train']))
            self.tokenizer.to_file(self._cachedir + '/vocab/snli.vocab')
        else:
            self.tokenizer.from_file(self._cachedir + '/vocab/snli.vocab')

        # Encode data
        if not path.exists(self._cachedir + '/encoded/snli.pkl'):
            os.makedirs(self._cachedir + '/encoded', exist_ok=True)

            data = {}
            for zipfile_name in ['train', 'dev', 'test']:
                data[zipfile_name] = [{
                    'sentence': self.tokenizer.encode(x['premise']),
                    'hypothesis': self.tokenizer.encode(x['hypothesis']),
                    'label': self.label_names.index(x['label']),
                } for x in dataset[zipfile_name]]

            with open(self._cachedir + '/encoded/snli.pkl', 'wb') as fp:
                pickle.dump(data, fp)

    def setup(self, stage=None):
        with open(self._cachedir + '/encoded/snli.pkl', 'rb') as fp:
            snli_dataset = pickle.load(fp)
        if stage == "fit":
            self._train = self._process_data(snli_dataset['train'])
            self._val = self._process_data(snli_dataset['dev'])
        elif stage == 'test':
            self._test = self._process_data(snli_dataset['test'])
        else:
            raise ValueError(f'unexpected setup stage: {stage}')

    def _process_data(self, data):
        return [{
            'sentence': torch.tensor(x['sentence'], dtype=torch.int64),
            'length': len(x['sentence']),
            'mask': torch.tensor(self.tokenizer.mask(x['sentence']), dtype=torch.bool),
            'hypothesis': torch.tensor(x['hypothesis'], dtype=torch.int64),
            'hypothesis_length': len(x['hypothesis']),
            'hypothesis_mask': torch.tensor(self.tokenizer.mask(x['hypothesis']), dtype=torch.bool),
            'label': torch.tensor(x['label'], dtype=torch.int64),
            'index': torch.tensor(idx, dtype=torch.int64)
        } for idx, x in enumerate(data)]

    def collate(self, observations):
        return {
            'sentence': self.tokenizer.stack_pad([observation['sentence'] for observation in observations]),
            'length': [observation['length'] for observation in observations],
            'mask': self.tokenizer.stack_pad([observation['mask'] for observation in observations]),
            'hypothesis': self.tokenizer.stack_pad([observation['hypothesis'] for observation in observations]),
            'hypothesis_length': [observation['hypothesis_length'] for observation in observations],
            'hypothesis_mask': self.tokenizer.stack_pad([observation['hypothesis_mask'] for observation in observations]),
            'label': torch.stack([observation['label'] for observation in observations]),
            'index': torch.stack([observation['index'] for observation in observations])
        }

    def uncollate(self, batch):
        return [{
            'sentence': sentence[:length],
            'mask': mask[:length],
            'length': length,
            'hypothesis': hypothesis[:hypothesis_length],
            'hypothesis_mask': hypothesis_mask[:hypothesis_length],
            'hypothesis_length': hypothesis_length,
            'label': label,
            'index': index
        } for sentence, mask, length,
              hypothesis, hypothesis_mask, hypothesis_length,
              label, index
          in zip(batch['sentence'], batch['mask'], batch['length'],
                 batch['hypothesis'], batch['hypothesis_mask'], batch['hypothesis_length'],
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
