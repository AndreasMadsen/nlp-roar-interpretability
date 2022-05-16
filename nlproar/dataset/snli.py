
import json
import pickle
import re
import os
import os.path as path
import requests
from io import BytesIO
from zipfile import ZipFile
from itertools import chain

import numpy as np
import spacy
import torchtext

from ._choose_tokenizer import choose_tokenizer
from ._vocab_tokenizer import VocabTokenizer
from ._paired_sequence_dataset import PairedSequenceDataset

class SNLITokenizer(VocabTokenizer):
    def __init__(self):
        # Original implementation has "en", we are using "en_core_web_sm"
        # https://github.com/successar/AttentionExplanation/blob/master/preprocess/vectorizer.py
        # https://github.com/successar/AttentionExplanation/blob/master/preprocess/SNLI/SNLI.ipynb
        super().__init__(min_df=3)
        self._tokenizer = spacy.load('en_core_web_sm',
                                     disable=['parser', 'tagger', 'ner', 'lemmatizer'])

    def tokenize(self, sentence):
        sentence = re.sub(r"\s+", " ", sentence.strip())
        sentence = [t.text.lower() for t in self._tokenizer(sentence)]
        sentence = [self.digits_token if any(char.isdigit()
                                             for char in word) else word for word in sentence]
        return sentence


class SNLIDataset(PairedSequenceDataset):
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
    def __init__(self, cachedir, model_type, batch_size=128, **kwargs):
        """Creates an SNLI dataset instance

        Args:
            cachedir (str): Directory to use for caching the compiled dataset.
            seed (int): Seed used for shuffling the dataset.
            batch_size (int, optional): The batch size used in the data loader. Defaults to 32.
            num_workers (int, optional): The number of pytorch workers in the data loader. Defaults to 4.
        """
        tokenizer = choose_tokenizer(cachedir, model_type, SNLITokenizer)
        super().__init__(cachedir, 'snli', model_type, tokenizer, batch_size=batch_size, **kwargs)
        self.label_names = ['entailment', 'contradiction', 'neutral']

    def embedding(self):
        """Creates word embedding matrix.

        Returns:
            np.array: shape = (vocabulary, 300)
        """
        if self.model_type != 'rnn':
            return None

        lookup = torchtext.vocab.pretrained_aliases['glove.840B.300d'](
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
        # In the Hugging Face distribution of the dataset,
        # the label has 4 possible values, 0, 1, 2, -1.
        # which correspond to entailment, neutral, contradiction,
        # and no label respectively.
        # https://github.com/huggingface/datasets/tree/master/datasets/snli
        torchtext.vocab.pretrained_aliases['glove.840B.300d'](
            cache=f'{self._cachedir}/embeddings')

        if (path.exists(f'{self._cachedir}/vocab/snli.vocab') and
            path.exists(f'{self._cachedir}/encoded/snli_{self.model_type}.pkl')):
            self.tokenizer.from_file(f'{self._cachedir}/vocab/snli.vocab')
            return

        # Download and parse data
        if not path.exists(f'{self._cachedir}/datasets/snli/snli_1.0.zip'):
            with open(f'{self._cachedir}/datasets/snli/snli_1.0.zip', 'wb') as fp:
                download_url = 'https://nlp.stanford.edu/projects/snli/snli_1.0.zip'
                fp.write(requests.get(download_url).content)

        dataset = {}
        # Change in data statistics on removing "no label"
        # Train - 550,152 -> 549367
        # Valid - 10,000 -> 9842
        # Test - 10,000 -> 9824

        with open(f'{self._cachedir}/datasets/snli/snli_1.0.zip', 'rb') as fp:
            zf = ZipFile(fp)

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
        if not path.exists(f'{self._cachedir}/vocab/snli.vocab'):
            print('building vocab')
            os.makedirs(f'{self._cachedir}/vocab', exist_ok=True)

            self.tokenizer.from_iterable(chain.from_iterable(
                (row['premise'], row["hypothesis"]) for row in dataset['train']))
            self.tokenizer.to_file(f'{self._cachedir}/vocab/snli.vocab')
        else:
            self.tokenizer.from_file(f'{self._cachedir}/vocab/snli.vocab')

        # Encode data
        if not path.exists(f'{self._cachedir}/encoded/snli_{self.model_type}.pkl'):
            os.makedirs(f'{self._cachedir}/encoded', exist_ok=True)

            data = {}
            for picklefile_name, zipfile_name in [('train', 'train'), ('val', 'dev'), ('test', 'test')]:
                data[picklefile_name] = [{
                    'sentence': self.tokenizer.encode(x['premise']),
                    'sentence_aux': self.tokenizer.encode(x['hypothesis']),
                    'label': self.label_names.index(x['label']),
                } for x in dataset[zipfile_name]]

            with open(f'{self._cachedir}/encoded/snli_{self.model_type}.pkl', 'wb') as fp:
                pickle.dump(data, fp)
