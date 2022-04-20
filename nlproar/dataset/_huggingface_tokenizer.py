
import torch
from transformers import AutoTokenizer

class HuggingfaceTokenizer:
    def __init__(self, cachedir, model_name):
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=f'{cachedir}/huggingface/tokenizer',
            use_fast=True
        )

        self.token_to_ids = self._tokenizer.vocab
        ids_to_token = { token_id: token for token, token_id in self.token_to_ids.items() }
        self.ids_to_token = [ids_to_token[token_id] for token_id in range(len(self.token_to_ids))]

        self.pad_token = self._tokenizer.pad_token
        self.pad_token_id = self._tokenizer.pad_token_id
        self.start_token = self._tokenizer.cls_token
        self.start_token_id = self._tokenizer.cls_token_id
        self.end_token = self._tokenizer.sep_token
        self.end_token_id = self._tokenizer.sep_token_id
        self.mask_token = self._tokenizer.mask_token
        self.mask_token_id = self._tokenizer.mask_token_id
        self.unknown_token = self._tokenizer.unk_token
        self.unknown_token_id = self._tokenizer.unk_token_id
        self.special_symbols = [
            self.pad_token,
            self.start_token, self.end_token,
            self.mask_token, self.unknown_token
        ]
        self.masked_tokens = {self.pad_token_id, self.start_token_id, self.end_token_id}

    def from_file(self, filepath):
        pass

    def to_file(self, filepath):
        pass

    def from_iterable(self, iterable):
        pass

    def tokenize(self, sentence):
        raise NotImplementedError()

    def encode(self, sentence):
        return self._tokenizer(sentence, return_attention_mask=False)['input_ids']

    def decode(self, token_ids):
        return tokenizer.decode(token_ids)

    def mask(self, token_ids):
        return [token_id not in self.masked_tokens for token_id in token_ids]

    def stack_pad(self, observations):
        max_length = max(tokens.shape[0] for tokens in observations)
        padded_observations = [
            torch.cat([tokens, torch.full(
                (max_length - tokens.shape[0], ), self.pad_token_id, dtype=tokens.dtype)], dim=0)
            for tokens in observations
        ]
        return torch.stack(padded_observations)

    def stack_pad_mask(self, observations):
        max_length = max(tokens.shape[0] for tokens in observations)
        padded_observations = [
            torch.cat([tokens, torch.zeros(
                max_length - tokens.shape[0], dtype=tokens.dtype)], dim=0)
            for tokens in observations
        ]
        return torch.stack(padded_observations)

class RobertaTokenizer(HuggingfaceTokenizer):
    def __init__(self, cachedir):
        super().__init__(cachedir, 'roberta-base')
        self.max_sequence_length = 512

    def truncate(self, sequence):
        if len(sequence) > self.max_sequence_length:
            sequence = sequence[:self.max_sequence_length - 1] + sequence[-1:]
        return sequence

    def sentence_pair(self, sentence, sentence_aux):
        return sentence + sentence_aux[1:]

    def sentence_type(self, sentence, sentence_aux=[0]):
        return [0] * len(sentence) + [1] * (len(sentence_aux) - 1)

class LongformerTokenizer(HuggingfaceTokenizer):
    def __init__(self, cachedir):
        super().__init__(cachedir, 'allenai/longformer-base-4096')
        self.max_sequence_length = 4096

    def truncate(self, sequence):
        if len(sequence) > self.max_sequence_length:
            sequence = sequence[:self.max_sequence_length - 1] + sequence[-1:]
        return sequence

    def sentence_pair(self, sentence, sentence_aux):
        return sentence + sentence_aux[1:]

    def sentence_type(self, sentence, sentence_aux=[0]):
        return [0] * len(sentence) + [1] * (len(sentence_aux) - 1)

class XLNetTokenizer(HuggingfaceTokenizer):
    def __init__(self, cachedir):
        super().__init__(cachedir, 'xlnet-base-cased')
        self.max_sequence_length = 2048 # XLNet runs out of memory above this

    def truncate(self, sequence):
        if len(sequence) > self.max_sequence_length:
            sequence = sequence[:self.max_sequence_length - 2] + sequence[-2:]
        return sequence

    def sentence_pair(self, sentence, sentence_aux):
        return sentence[:-1] + sentence_aux

    def sentence_type(self, sentence, sentence_aux=[0]):
        return [0] * (len(sentence) - 1) + [1] * (len(sentence_aux) - 1) + [2]
