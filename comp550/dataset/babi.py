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

from .tokenizer import Tokenizer


IDX_TO_TASK_MAP = {1: 'qa1_single-supporting-fact_',
                   2: 'qa2_two-supporting-facts_',
                   3: 'qa3_three-supporting-facts_'}


class BabiTokenizer(Tokenizer):
    def __init__(self):

        super().__init__()
        self.min_df = 1

    def tokenize(self, sentence):
        return sentence.split()


class BabiDataModule(pl.LightningDataModule):

    def __init__(self, cachedir, batch_size=50, num_workers=4, task=1):
        super().__init__()
        self._cachedir = cachedir
        self.batch_size = batch_size
        self._num_workers = num_workers
        self._task = task

        self.tokenizer = BabiTokenizer()
        self.name = f'babi-{task}'

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
        for word in self.tokenizer.ids_to_token:
            if word == self.tokenizer.pad_token:
                embeddings.append(np.zeros(50))
            else:
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
        with open(file, 'r', encoding='utf-8') as fp:
            for line in fp:
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

        os.makedirs(self._cachedir + '/text-datasets', exist_ok=True)
        if not path.exists(self._cachedir + '/text-datasets/tasks_1-20_v1-2'):
            r = requests.get(babi_url, stream=True)
            with open(self._cachedir + '/text-datasets/tasks_1-20_v1-2.tar.gz', 'wb') as f:
                f.write(r.raw.read())

            with open(self._cachedir + '/text-datasets/tasks_1-20_v1-2.tar.gz', 'rb') as f:
                thetarfile = tarfile.open(fileobj=f, mode="r|gz")
                thetarfile.extractall(path=self._cachedir + '/text-datasets')
                thetarfile.close()


        if not path.exists(self._cachedir + '/text-datasets/babi.pkl'):

            tasks = IDX_TO_TASK_MAP.keys()
            data = {}
            for t in tasks:
                data[t] = {}
                for k in ['train', 'test']:
                    data[t][k] = self._parse(
                        self._cachedir + '/text-datasets/tasks_1-20_v1-2/en-10k/' + IDX_TO_TASK_MAP[t] + k + '.txt')

            with open(self._cachedir + '/text-datasets/babi.pkl', 'wb') as fp:
                pickle.dump(data, fp)

        else:
            with open(self._cachedir + '/text-datasets/babi.pkl', 'rb') as fp:
                data = pickle.load(fp)

        if not path.exists(self._cachedir + '/output-labels/babi.label'):
            os.makedirs(self._cachedir + '/output-labels', exist_ok=True)

            output_labels = self._get_output_labels(data)
            with open(self._cachedir + '/output-labels/babi.label', 'w', encoding='utf-8') as fp:
                for token in output_labels:
                    print(token, file=fp)

        else:
            with open(self._cachedir + '/output-labels/babi.label', 'r', encoding='utf-8') as fp:
                output_labels = [line.strip() for line in fp]

        self.label_names = output_labels

        if not path.exists(self._cachedir + f'/text-datasets/{self.name}.pkl'):
            trainidx, devidx = train_test_split(
                range(0, len(data[self._task]['train'])), train_size=0.85)
            testidx = range(0, len(data[self._task]['test']))

            babi_data = {}
            for name, idxs, dataset in [('train', trainidx, data[self._task]['train']), ('val', devidx, data[self._task]['train']), ('test', testidx, data[self._task]['test'])]:
                babi_data[name] = [{
                    'paragraph': dataset[idx]["paragraph"],
                    'question': dataset[idx]["question"],
                    'label': self.label_names.index(dataset[idx]["answer"])
                } for idx in idxs]

            with open(self._cachedir + f'/text-datasets/{self.name}.pkl', 'wb') as fp:
                pickle.dump(babi_data, fp)

        else:
            with open(self._cachedir + f'/text-datasets/{self.name}.pkl', 'rb') as fp:
                babi_data = pickle.load(fp)


        if not path.exists(self._cachedir + f'/vocab/babi_{self._task}.vocab'):
            os.makedirs(self._cachedir + '/vocab', exist_ok=True)

            self.tokenizer.from_iterable(
                [instance["paragraph"] for instance in babi_data['train']] +
                [instance["question"] for instance in babi_data['train']])
            self.tokenizer.to_file(
                self._cachedir + f'/vocab/babi_{self._task}.vocab')
        else:
            self.tokenizer.from_file(
                self._cachedir + f'/vocab/babi_{self._task}.vocab')


        if not path.exists(self._cachedir + f'/encoded/{self.name}.pkl'):
            os.makedirs(self._cachedir + '/encoded', exist_ok=True)

            data = {}
            for name in ['train', 'val', 'test']:

                dataset = babi_data[name]
                data[name] = [{
                    'paragraph': self.tokenizer.encode(instance["paragraph"]),
                    'question': self.tokenizer.encode(instance["question"]),
                    'label': instance['label']
                } for instance in dataset]

            with open(self._cachedir + f'/encoded/{self.name}.pkl', 'wb') as fp:
                pickle.dump(data, fp)

    def setup(self, stage=None):
        with open(self._cachedir + f'/encoded/{self.name}.pkl', 'rb') as fp:
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

    def train_dataloader(self, batch_size=None):
        return DataLoader(self._train,
                          batch_size=batch_size or self.batch_size, collate_fn=self.collate,
                          num_workers=self._num_workers, shuffle=True)

    def val_dataloader(self, batch_size=None):
        return DataLoader(self._val,
                          batch_size=batch_size or self.batch_size, collate_fn=self.collate,
                          num_workers=self._num_workers)

    def test_dataloader(self, batch_size=None):
        return DataLoader(self._test,
                          batch_size=batch_size or self.batch_size, collate_fn=self.collate,
                          num_workers=self._num_workers)

