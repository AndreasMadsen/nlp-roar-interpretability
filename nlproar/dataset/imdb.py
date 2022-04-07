import os.path as path
import os
import torch
import pickle
import json
import requests

import numpy as np
import torchtext
from sklearn.model_selection import train_test_split

from ._roberta_tokenizer import RobertaTokenizer
from ._vocab_tokenizer import VocabTokenizer
from ._single_sequence_dataset import SingleSequenceDataset


class IMDBTokenizer(VocabTokenizer):
    def __init__(self):
        # Document frequency is 10
        # https://github.com/successar/AttentionExplanation/blob/master/preprocess/IMDB/IMDB.ipynb
        super().__init__(min_df=10)

    def tokenize(self, sentence):
        return sentence.split()


class IMDBDataset(SingleSequenceDataset):
    def __init__(self, cachedir, model_type, batch_size=32, **kwargs):
        """Creates an IMDB dataset instance

        Args:
            cachedir (str): Directory to use for caching the compiled dataset.
            seed (int): Seed used for shuffling the dataset.
            batch_size (int, optional): The batch size used in the data loader. Defaults to 32.
            num_workers (int, optional): The number of pytorch workers in the data loader. Defaults to 4.
        """
        tokenizer = RobertaTokenizer(cachedir) if model_type == 'roberta' else IMDBTokenizer()
        super().__init__(cachedir, 'imdb', model_type, tokenizer, batch_size=batch_size, **kwargs)
        self.label_names = ['negative', 'positive']

    def embedding(self):
        """Creates word embedding matrix.

        Returns:
            np.array: shape = (vocabulary, 300)
        """
        lookup = torchtext.vocab.pretrained_aliases['fasttext.simple.300d'](
            cache=f'{self._cachedir}/embeddings')

        embeddings = []
        for word in self.tokenizer.ids_to_token:
            if word in set(self.tokenizer.special_symbols) or word not in lookup.stoi:
                embeddings.append(np.zeros(300))
            else:
                embeddings.append(lookup[word].numpy())

        return np.vstack(embeddings)

    def prepare_data(self):
        """Download, compiles, and cache the dataset.
        """
        torchtext.vocab.pretrained_aliases['fasttext.simple.300d'](
            cache=f'{self._cachedir}/embeddings')

        # Short-circuit the build logic if the minimum-required files exists
        if (path.exists(f'{self._cachedir}/encoded/imdb.pkl') and
            path.exists(f'{self._cachedir}/vocab/imdb.vocab')):
            self.tokenizer.from_file(f'{self._cachedir}/vocab/imdb.vocab')
            return

        # Download data
        imdb_data_s3_url = 'https://s3.amazonaws.com/text-datasets/imdb_full.pkl'
        os.makedirs(f'{self._cachedir}/text-datasets', exist_ok=True)
        if not path.exists(f'{self._cachedir}/text-datasets/imdb_full.pkl'):
            r = requests.get(imdb_data_s3_url)
            with open(f'{self._cachedir}/text-datasets/imdb_full.pkl', 'wb') as f:
                f.write(r.content)

        imdb_vocab_s3_url = 'https://s3.amazonaws.com/text-datasets/imdb_word_index.json'
        if not path.exists(f'{self._cachedir}/text-datasets/imdb_word_index.json'):
            r = requests.get(imdb_vocab_s3_url)
            with open(f'{self._cachedir}/text-datasets/imdb_word_index.json', 'wb') as f:
                f.write(r.content)

        # Build dataset
        with open(f'{self._cachedir}/text-datasets/imdb_full.pkl', 'rb') as fp:
            train_set, test_set = pickle.load(fp)

        with open(f'{self._cachedir}/text-datasets/imdb_word_index.json', 'rb') as fp:
            idx_to_word_map = {idx:word for word, idx in json.load(fp).items()}

        # In the original implementation:
        #  * Sentences with length greater than or equal to 400 are removed
        #  * Only 20% of test set is used https://github.com/successar/AttentionExplanation/blob/master/preprocess/IMDB/IMDB.ipynb
        trainidx = [i for i, x in enumerate(train_set[0]) if len(x) < 400]
        trainidx, devidx = train_test_split(trainidx, train_size=0.8, random_state=self._np_rng)
        testidx = [i for i, x in enumerate(test_set[0]) if len(x) < 400]
        testidx, _ = train_test_split(testidx, train_size=0.2, random_state=self._np_rng)

        imdb_data = {}
        for name, idxs, dataset in [('train', trainidx, train_set), ('val', devidx, train_set), ('test', testidx, test_set)]:
            # Min length is 6
            # https://github.com/successar/AttentionExplanation/blob/425a89a49a8b3bffc3f5e8338287e2ecd0cf1fa2/Trainers/DatasetBC.py#L108
            imdb_data[name] = [{
                'sentence': " ".join([idx_to_word_map[x] for x in dataset[0][idx]]),
                'label': dataset[1][idx]
            } for idx in idxs if len(dataset[0][idx]) > 6]

        # Build vocabulary
        if not path.exists(f'{self._cachedir}/vocab/imdb.vocab'):
            os.makedirs(f'{self._cachedir}/vocab', exist_ok=True)

            self.tokenizer.from_iterable(x['sentence'] for x in imdb_data['train'])
            self.tokenizer.to_file(f'{self._cachedir}/vocab/imdb.vocab')
        else:
            self.tokenizer.from_file(f'{self._cachedir}/vocab/imdb.vocab')

        # Encode dataset
        if not path.exists(f'{self._cachedir}/encoded/imdb_{self.model_type}.pkl'):
            os.makedirs(f'{self._cachedir}/encoded', exist_ok=True)

            data = {}
            for name in ['train', 'val', 'test']:

                dataset = imdb_data[name]
                data[name] = [{
                    'sentence': self.tokenizer.encode(instance['sentence']),
                    'label': instance['label']
                } for instance in dataset]

            with open(f'{self._cachedir}/encoded/imdb_{self.model_type}.pkl', 'wb') as fp:
                pickle.dump(data, fp)
