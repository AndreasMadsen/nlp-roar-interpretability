
from comp550.dataset.tokenizer import Tokenizer
import os.path as path
import re
import os
import random
import pickle
import warnings
import requests
from multiprocessing import Pool
from functools import partial

import torchtext
import torch
import spacy
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from gensim.models import Word2Vec, KeyedVectors
from tqdm import tqdm

from .tokenizer import Tokenizer

class _Normalizer(Tokenizer):
    def normalize(self, sentence):
        sentence = re.sub(r'\[\s*\*\s*\*(.*?)\*\s*\*\s*\]', ' <DE> ', sentence)
        sentence = re.sub(r'([^a-zA-Z0-9])(\s*\1\s*)+', r'\1 ', sentence)
        sentence = re.sub(r'\s+', ' ', sentence.strip())
        sentence = [t.text.lower() for t in self._tokenizer(sentence)]
        sentence = [
            self.digits_token if any(char.isdigit() for char in word) else word
            for word in sentence
        ]
        return ' '.join(sentence)

class MimicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
        self.min_df = 5

    def tokenize(self, sentence):
        return sentence.split(' ')

class MimicDataset(pl.LightningDataModule):
    """Loads the Diabetes or Anemia dataset from MIMIC (III)

    Note that there are special min-max lengths, see:
        https://github.com/successar/AttentionExplanation/blob/master/Trainers/DatasetBC.py#L113
    """
    def __init__(self, cachedir, mimicdir, batch_size=32, seed=0, num_workers=4, subset='diabetes'):
        super().__init__()
        if subset not in ['diabetes', 'anemia']:
            raise ValueError('subset must be either "diabetes" or "anemia"')

        self._cachedir = path.realpath(cachedir)
        self._mimicdir = path.realpath(mimicdir)
        self.batch_size = batch_size
        self._seed = seed
        self._num_workers = num_workers
        self._subset = subset

        self.tokenizer = MimicTokenizer()
        self.label_names = ['negative', 'positive']
        self.name = f'mimic-{subset[0]}'

    @property
    def vocabulary(self):
        return self.tokenizer.ids_to_token

    def embedding(self):
        lookup = KeyedVectors.load(f'{self._cachedir}/embeddings/mimic.wv')
        rng = np.random.RandomState(self._seed)

        embeddings = []
        for word in self.tokenizer.ids_to_token:
            if word == self.tokenizer.pad_token:
                embeddings.append(np.zeros(300))
            if word == self.tokenizer.digits_token:
                embeddings.append(lookup[word])
            if word in set(self.tokenizer.special_symbols) or word not in lookup:
                embeddings.append(rng.randn(300))
            else:
                embeddings.append(lookup[word])

        return np.vstack(embeddings)

    def prepare_data(self):
        # Short-circuit the build logic if the minimum-required files exists
        if (path.exists(f'{self._cachedir}/embeddings/mimic.wv') and
            path.exists(self._cachedir + '/vocab/mimic_anemia.vocab') and
            path.exists(self._cachedir + '/vocab/mimic_diabetes.vocab') and
            path.exists(f'{self._cachedir}/encoded/mimic_anemia.pkl') and
            path.exists(f'{self._cachedir}/encoded/mimic_diabetes.pkl')):
            self.tokenizer.from_file(self._cachedir + f'/vocab/mimic_{self._subset}.vocab')
            return

        # Ensure that confidential files exists
        if not path.exists(f'{self._mimicdir}/DIAGNOSES_ICD.csv.gz'):
            raise IOError(f'The file "{self._mimicdir}/DIAGNOSES_ICD.csv.gz" is missing')
        if not path.exists(f'{self._mimicdir}/NOTEEVENTS.csv.gz'):
            raise IOError(f'The file "{self._mimicdir}/NOTEEVENTS.csv.gz" is missing')

        # Download splitfiles
        os.makedirs(f'{self._cachedir}/mimic-dataset/hadm_ids', exist_ok=True)
        for split in ['train', 'dev', 'test']:
            if not path.exists(f'{self._cachedir}/mimic-dataset/hadm_ids/{split}_50_hadm_ids.csv'):
                with open(f'{self._cachedir}/mimic-dataset/hadm_ids/{split}_50_hadm_ids.csv', 'wb') as fp:
                    download_url = ('https://raw.githubusercontent.com/successar/AttentionExplanation'
                                   f'/master/preprocess/MIMIC/{split}_50_hadm_ids.csv')
                    fp.write(requests.get(download_url).content)

        # Build merged and normalized datafile
        df_merged = None
        if not path.exists(f'{self._cachedir}/mimic-dataset/merged.csv.gz'):
            # Filter and collect ICD9-codes for each subject+HADM
            df_icd9_codes = pd.read_csv(f'{self._mimicdir}/DIAGNOSES_ICD.csv.gz', compression='gzip',
                                        usecols=['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE'])
            df_icd9_codes.dropna(inplace=True)
            df_icd9_codes = df_icd9_codes.groupby(['SUBJECT_ID', 'HADM_ID'], as_index=False).agg({
                'ICD9_CODE': lambda codes: ';'.join(code[:3] + '.' + code[3:] for code in codes)
            })

            # Filter and collect discharge summaries
            print('Reading MIMIC CSV file')
            df_notes = pd.read_csv(f'{self._mimicdir}/NOTEEVENTS.csv.gz', compression='gzip',
                                usecols=['SUBJECT_ID', 'HADM_ID', 'CATEGORY', 'CHARTDATE', 'DESCRIPTION', 'TEXT'])
            df_notes.dropna(inplace=True)
            df_notes = df_notes[df_notes['CATEGORY'] == 'Discharge summary']
            df_notes.replace({'DESCRIPTION': {'Report' : 0, 'Addendum' : 1}}, inplace=True)
            df_notes.sort_values(by=['DESCRIPTION', 'CHARTDATE'], inplace=True)
            df_notes = df_notes.groupby(['SUBJECT_ID', 'HADM_ID'], as_index=False).agg({
                'TEXT': lambda texts: " ".join(texts).strip()
            })

            # Merge tables
            print('Merging MIMIC')
            df_merged = df_notes.merge(df_icd9_codes, on=['SUBJECT_ID', 'HADM_ID'])

            # Clean data
            print('Cleaning MIMIC')
            with Pool(processes=self._num_workers) as p:
                df_merged['TEXT'] = p.map(_Normalizer().normalize, df_merged['TEXT'])
            df_merged.to_csv(f'{self._cachedir}/mimic-dataset/merged.csv.gz', compression='gzip', index=False)

        # Build embedding model
        os.makedirs(f'{self._cachedir}/embeddings', exist_ok=True)
        if not path.exists(f'{self._cachedir}/embeddings/mimic.wv'):
            print('building embedding model')
            if df_merged is None:
                df_merged = pd.read_csv(f'{self._cachedir}/mimic-dataset/merged.csv.gz', compression='gzip')

            embedding = Word2Vec(map(lambda x: x.split(' '), df_merged['TEXT']),
                            size=300, window=10, min_count=2,
                            workers=self._num_workers)
            embedding.wv.save(f'{self._cachedir}/embeddings/mimic.wv')

        # Build anemia dataset
        os.makedirs(f'{self._cachedir}/encoded', exist_ok=True)
        if not path.exists(f'{self._cachedir}/encoded/mimic_anemia.pkl'):
            print('building anemia dataset')
            if df_merged is None:
                df_merged = pd.read_csv(f'{self._cachedir}/mimic-dataset/merged.csv.gz', compression='gzip')

            # Filter data and assign target
            codes = df_merged['ICD9_CODE'].str.split(';')
            has_c1 = codes.apply(lambda x: any(code.startswith('285.1') for code in x))
            has_c2 = codes.apply(lambda x: any(code.startswith('285.2') for code in x))
            df_anemia = df_merged.loc[has_c1 ^ has_c2, :]
            df_anemia = df_anemia.assign(
                target = has_c1[has_c1 ^ has_c2].astype('int64')
            )

            # Split data
            all_idx = list(range(len(df_anemia)))
            train_idx, test_idx = train_test_split(all_idx, stratify=df_anemia.loc[:, 'target'],
                                                   test_size=0.2, random_state=12939)
            train_idx, val_idx = train_test_split(train_idx, stratify=df_anemia.loc[:, 'target'].iloc[train_idx],
                                                  test_size=0.15, random_state=13448)

            # Build vocabulary
            os.makedirs(self._cachedir + '/vocab', exist_ok=True)
            tokenizer_anemia = MimicTokenizer()
            tokenizer_anemia.from_iterable(df_anemia.iloc[train_idx, :].loc[:, 'TEXT'])
            tokenizer_anemia.to_file(self._cachedir + '/vocab/mimic_anemia.vocab')

            # Encode dataset
            data_anemia = {}
            for name, split in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
                observations = []
                for idx in split:
                    token_idx = tokenizer_anemia.encode(df_anemia.loc[:, 'TEXT'].iat[idx])
                    if len(token_idx) - 2 > 4000:
                        continue
                    observations.append({
                        'sentence': token_idx,
                        'label': df_anemia.loc[:, 'target'].iat[idx],
                        'index': idx
                    })
                data_anemia[name] = observations

            # Save dataset
            with open(self._cachedir + '/encoded/mimic_anemia.pkl', 'wb') as fp:
                pickle.dump(data_anemia, fp)

        # Build diabetes dataset
        os.makedirs(f'{self._cachedir}/encoded', exist_ok=True)
        if not path.exists(f'{self._cachedir}/encoded/mimic_diabetes.pkl'):
            print('building diabetes dataset')
            if df_merged is None:
                df_merged = pd.read_csv(f'{self._cachedir}/mimic-dataset/merged.csv.gz', compression='gzip')

            # Load predefied hadm_ids
            hadm_ids = {}
            for split in ['train', 'dev', 'test']:
                hadm_ids_df = pd.read_csv(f'{self._cachedir}/mimic-dataset/hadm_ids/{split}_50_hadm_ids.csv', header=None)
                hadm_ids[split] = list(hadm_ids_df[0])
            hadm_ids_all = hadm_ids['train'] + hadm_ids['dev'] + hadm_ids['test']

            # Filter data and assign target
            df_diabetes = df_merged.loc[df_merged['HADM_ID'].isin(hadm_ids_all), :]
            codes = df_diabetes['ICD9_CODE'].str.split(';')
            has_d1 = codes.apply(lambda x: any(code.startswith('250.00') for code in x))
            df_diabetes = df_diabetes.assign(
                target = has_d1.astype('int64'),
                index = np.arange(len(df_diabetes))
            )

            # Build vocabulary
            os.makedirs(self._cachedir + '/vocab', exist_ok=True)
            tokenizer_diabetes = MimicTokenizer()
            tokenizer_diabetes.from_iterable(df_diabetes.loc[df_diabetes['HADM_ID'].isin(hadm_ids['train']), 'TEXT'])
            tokenizer_diabetes.to_file(self._cachedir + '/vocab/mimic_diabetes.vocab')

            # Encode dataset
            data_diabetes = {}
            for name, split in [('train', hadm_ids['train']), ('val', hadm_ids['dev']), ('test', hadm_ids['test'])]:
                df_split = df_diabetes.loc[df_diabetes['HADM_ID'].isin(split), :]
                observations = []
                for idx in range(len(df_split)):
                    token_idx = tokenizer_diabetes.encode(df_split.loc[:, 'TEXT'].iat[idx])
                    if 6 > len(token_idx) - 2 > 4000:
                        continue
                    observations.append({
                        'sentence': token_idx,
                        'label': df_split.loc[:, 'target'].iat[idx],
                        'index': df_split.loc[:, 'index'].iat[idx]
                    })
                data_diabetes[name] = observations

            # Save dataset
            with open(self._cachedir + '/encoded/mimic_diabetes.pkl', 'wb') as fp:
                pickle.dump(data_diabetes, fp)

        # Load relevant vocabulary
        self.tokenizer.from_file(self._cachedir + f'/vocab/mimic_{self._subset}.vocab')

    def _process_data(self, data):
        return [{
            'sentence': torch.tensor(x['sentence'], dtype=torch.int64),
            'mask': torch.tensor(self.tokenizer.mask(x['sentence']), dtype=torch.bool),
            'length': len(x['sentence']),
            'label': torch.tensor(x['label'], dtype=torch.int64),
            'index': torch.tensor(x['index'], dtype=torch.int64)
        } for x in data]

    def setup(self, stage=None):
        with open(self._cachedir + f'/encoded/mimic_{self._subset}.pkl', 'rb') as fp:
            data = pickle.load(fp)
        if stage == 'fit':
            self._train = self._process_data(data['train'])
            self._val = self._process_data(data['val'])
        elif stage == 'test':
            self._test = self._process_data(data['test'])
        else:
            raise ValueError(f'unexpected setup stage: {stage}')

    def collate(self, observations):
        return {
            'sentence': self.tokenizer.stack_pad([observation['sentence'] for observation in observations]),
            'mask': self.tokenizer.stack_pad([observation['mask'] for observation in observations]),
            'length': [observation['length'] for observation in observations],
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
