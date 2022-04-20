
from ._huggingface_tokenizer import RobertaTokenizer, LongformerTokenizer, XLNetTokenizer

def choose_tokenizer(cachedir, model_type, RNNTokenizer):
    if model_type == 'roberta':
        return RobertaTokenizer(cachedir)
    elif model_type == 'longformer':
        return LongformerTokenizer(cachedir)
    elif model_type == 'xlnet':
        return XLNetTokenizer(cachedir)
    elif model_type == 'rnn':
        return RNNTokenizer()
