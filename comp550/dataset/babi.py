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
import tarfile

from torch.utils.data import Dataset


IDX_TO_TASK_MAP = {1: 'qa1_single-supporting-fact_',
                   2: 'qa2_two-supporting-facts_',
                   3: 'qa3_three-supporting-facts_'}


class _Tokenizer:
    def __init__(self):
        '''
        Original implementation has "en", we are using "en_core_web_sm"
        https://github.com/successar/AttentionExplanation/blob/master/preprocess/vectorizer.py
        https://github.com/successar/AttentionExplanation/blob/master/preprocess/SNLI/SNLI.ipynb
        '''
        self.ids_to_token = []
        self.token_to_ids = {}
        self.min_df = 1

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
        counter = Counter()
        for sentence in iterable:
            counter.update(set(sentence.split()))

        tokens = [x for x in counter.keys() if counter[x] >= self.min_df]
        self.ids_to_token = self.special_symbols + tokens
        self._update_token_to_ids()

    def tokenize(self, sentence):

        sentence = re.sub(r"\s+", " ", sentence.strip())
        sentence = [t.text.lower() for t in self._tokenizer(sentence)]
        sentence = [self.digits_token if any(char.isdigit()
                                             for char in word) else word for word in sentence]
        return sentence

    def encode(self, sentence):
        return [self.start_token_id] + [
            self.token_to_ids.get(word, self.unknown_token_id)
            for word in sentence.split()
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


class BabiDataModule(pl.LightningDataModule):

    def __init__(self, cachedir, batch_size=50, num_workers=4, task_idx=1):
        super().__init__()
        self._cachedir = cachedir
        self._batch_size = batch_size
        self._num_workers = num_workers
        self.tokenizer = _Tokenizer()
        self.task_idx = task_idx

    @property
    def vocabulary(self):
        return self.tokenizer.ids_to_token

    def embedding(self):
        '''
        Random embeddings of size 50
        https://github.com/successar/AttentionExplanation/blob/425a89a49a8b3bffc3f5e8338287e2ecd0cf1fa2/model/modules/Encoder.py#L26
        https://github.com/successar/AttentionExplanation/blob/425a89a49a8b3bffc3f5e8338287e2ecd0cf1fa2/Trainers/DatasetQA.py#L112
        '''
        embeddings = []
        for _ in self.tokenizer.ids_to_token:
            embeddings.append(np.random.randn(50))

        return np.vstack(embeddings)

    def _get_output_labels(self, data):

        labels = set()

        for task in data.keys():
            for instance in data[task]["train"]:
                for word in instance["paragraph"].split():
                    labels.add(word)
                for word in instance["question"].split():
                    labels.add(word)
        return list(labels)

    def _parse(self, file):
        data, story = [], []
        for line in open(file).readlines():
            tid, text = line.rstrip('\n').split(' ', 1)
            if tid == '1':
                story = []
            # sentence
            if text.endswith('.'):
                story.append(text[:-1])
            # question
            else:
                query, answer, _ = (x.strip() for x in text.split('\t'))
                substory = " . ".join([x for x in story if x])
                data.append(
                    {"paragraph": substory, "question": query[:-1], "answer": answer})
        return data

    def prepare_data(self):
        '''
        '''
        babi_url = 'http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz'


        if not path.exists(self._cachedir + '/text-datasets/tasks_1-20_v1-2'):
            r = requests.get(babi_url, stream=True)
            with open(self._cachedir + '/text-datasets/tasks_1-20_v1-2.tar.gz', 'wb') as f:
                f.write(r.raw.read())

            with open(self._cachedir + '/text-datasets/tasks_1-20_v1-2.tar.gz', 'rb') as f:
                thetarfile = tarfile.open(fileobj=f, mode="r|gz")
                thetarfile.extractall(path=self._cachedir + '/text-datasets')
                thetarfile.close()


        if not path.exists(self._cachedir + '/text-datasets/babi.pkl'):
            os.makedirs(self._cachedir + '/text-datasets', exist_ok=True)
            tasks = IDX_TO_TASK_MAP.keys()
            data = {}
            for t in tasks:
                data[t] = {}
                for k in ['train', 'test']:
                    data[t][k] = self._parse(
                        self._cachedir + '/text-datasets/tasks_1-20_v1-2/en-10k/' + IDX_TO_TASK_MAP[t] + k + '.txt')

            with open(self._cachedir + '/text-datasets/babi.pkl', 'wb') as fp:
                pickle.dump(data, fp)

            output_labels = self._get_output_labels(data)

            os.makedirs(self._cachedir + '/output-labels', exist_ok=True)
            with open(self._cachedir + '/output-labels/babi.label', 'w') as fp:
                for token in output_labels:
                    print(token, file=fp)

        else:
            with open(self._cachedir + '/text-datasets/babi.pkl', 'rb') as fp:
                data = pickle.load(fp)
            with open(self._cachedir + '/output-labels/babi.label', 'r') as fp:
                output_labels = [line.strip() for line in fp]

        output_labels = {token: id for id, token in enumerate(output_labels)}
        self.num_classes = len(output_labels.keys())

        if not path.exists(self._cachedir + f'/vocab/babi{self.task_idx}.vocab'):
            os.makedirs(self._cachedir + '/vocab', exist_ok=True)

            trainidx, devidx = train_test_split(
                range(0, len(data[self.task_idx]['train'])), train_size=0.85)

            self.tokenizer.from_iterable(
                [instance["paragraph"] for idx, instance in enumerate(data[self.task_idx]['train']) if idx in trainidx] +
                [instance["question"] for idx, instance in enumerate(data[self.task_idx]['train']) if idx in trainidx])
            self.tokenizer.to_file(
                self._cachedir + f'/vocab/babi{self.task_idx}.vocab')
        else:
            self.tokenizer.from_file(
                self._cachedir + f'/vocab/babi{self.task_idx}.vocab')

        if not path.exists(self._cachedir + f'/encoded/babi{self.task_idx}.pkl'):
            os.makedirs(self._cachedir + '/encoded', exist_ok=True)

            testidx = range(0, len(data[self.task_idx]['test']))

            babi_data = {}
            for name, idxs, dataset in [('train', trainidx, data[self.task_idx]['train']), ('val', devidx, data[self.task_idx]['train']), ('test', testidx, data[self.task_idx]['test'])]:

                babi_data[name] = [{
                    'paragraph': self.tokenizer.encode(dataset[idx]["paragraph"]),
                    'question': self.tokenizer.encode(dataset[idx]["question"]),
                    'label': output_labels[dataset[idx]["answer"]]
                } for idx in idxs]

            with open(self._cachedir + f'/encoded/babi{self.task_idx}.pkl', 'wb') as fp:
                pickle.dump(babi_data, fp)

    def setup(self, stage=None):
        with open(self._cachedir + f'/encoded/babi{self.task_idx}.pkl', 'rb') as fp:
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
            'sentence': torch.tensor(x['paragraph'], dtype=torch.int64),
            'length': len(x['paragraph']),
            'mask': torch.tensor(self.tokenizer.mask(x['paragraph']), dtype=torch.bool),
            'hypothesis': torch.tensor(x['question'], dtype=torch.int64),
            'hypothesis_length': len(x['question']),
            'hypothesis_mask': torch.tensor(self.tokenizer.mask(x['question']), dtype=torch.bool),
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
