import spacy
import torch
from collections import Counter

class Tokenizer:
    def __init__(self):

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
        with open(filepath, 'r', encoding='utf-8') as fp:
            self.ids_to_token = [line.strip() for line in fp]
        self._update_token_to_ids()

    def to_file(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as fp:
            for token in self.ids_to_token:
                print(token, file=fp)

    def from_iterable(self, iterable):
        counter = Counter()
        for sentence in iterable:
            counter.update(set(self.tokenize(sentence)))

        tokens = [x for x in counter.keys() if counter[x] >= self.min_df]
        self.ids_to_token = self.special_symbols + tokens
        self._update_token_to_ids()

    def tokenize(self, sentence):
        raise NotImplementedError()

    def encode(self, sentence):
        return [self.start_token_id] + [
            self.token_to_ids.get(word, self.unknown_token_id)
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