
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

from ._roberta_tokenizer import RobertaTokenizer
from ._vocab_tokenizer import VocabTokenizer
from ._single_sequence_dataset import SingleSequenceDataset

class SSTTokenizer(VocabTokenizer):
    def __init__(self):
        super().__init__()
        self._tokenizer = spacy.load('en_core_web_sm',
                                     disable=['parser', 'tagger', 'ner', 'lemmatizer'])

    def tokenize(self, sentence):
        sentence = sentence.strip()
        sentence = sentence.replace("-LRB-", '')
        sentence = sentence.replace("-RRB-", '  ')
        sentence = re.sub(r'\W+', ' ', sentence)
        sentence = re.sub(r'\s+', ' ', sentence)
        return [t.text.lower() for t in self._tokenizer(sentence)]


class SSTDataset(SingleSequenceDataset):
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
     * Removes sentences shorter than 5 (https://github.com/successar/AttentionExplanation/blob/master/Trainers/DatasetBC.py#L103)
     * Batch size of 32 (https://github.com/successar/AttentionExplanation/blob/master/configurations.py#L19)

    The paper's embedding code is in:
        https://github.com/successar/AttentionExplanation/blob/master/preprocess/vectorizer.py#L103
    In general:
    * use 'fasttext.simple.300d'
    * set [PAD] embedding to zero
    """
    def __init__(self, cachedir, model_type, seed=0, **kwargs):
        """Creates an SST dataset instance

        Args:
            cachedir (str): Directory to use for caching the compiled dataset.
            seed (int): Seed used for shuffling the dataset.
            batch_size (int, optional): The batch size used in the data loader. Defaults to 32.
            num_workers (int, optional): The number of pytorch workers in the data loader. Defaults to 4.
        """
        tokenizer = RobertaTokenizer(cachedir) if model_type == 'roberta' else SSTTokenizer()
        super().__init__(cachedir, 'sst', model_type, tokenizer, **kwargs)
        self.label_names = ['negative', 'positive']

    def embedding(self):
        """Creates word embedding matrix.

        Returns:
            np.array: shape = (vocabulary, 300)
        """
        lookup = torchtext.vocab.pretrained_aliases['fasttext.simple.300d'](cache=f'{self._cachedir}/embeddings')

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
        # Load embeddings
        torchtext.vocab.pretrained_aliases['fasttext.simple.300d'](cache=f'{self._cachedir}/embeddings')

        # Load dataset
        if (not path.exists(f'{self._cachedir}/vocab/sst.vocab') or
            not path.exists(f'{self._cachedir}/encoded/sst.pkl')):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                # SST has not been migrated to the new torchtext.datasets yet
                train, val, test = torchtext.legacy.datasets.SST.splits(
                    torchtext.legacy.data.Field(), torchtext.legacy.data.Field(sequential=False),
                    filter_pred=lambda ex: len(ex.text) > 5 and ex.label != 'neutral',
                    root=f'{self._cachedir}/datasets')

        # Create vocabulary from training data, if it hasn't already been done
        if not path.exists(f'{self._cachedir}/vocab/sst.vocab'):
            os.makedirs(f'{self._cachedir}/vocab', exist_ok=True)

            self.tokenizer.from_iterable(' '.join(row.text) for row in train)
            self.tokenizer.to_file(f'{self._cachedir}/vocab/sst.vocab')
        else:
            self.tokenizer.from_file(f'{self._cachedir}/vocab/sst.vocab')

        # Encode data
        if not path.exists(f'{self._cachedir}/encoded/sst_{self.model_type}.pkl'):
            os.makedirs(f'{self._cachedir}/encoded', exist_ok=True)

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

            with open(f'{self._cachedir}/encoded/sst_{self.model_type}.pkl', 'wb') as fp:
                pickle.dump(data, fp)
