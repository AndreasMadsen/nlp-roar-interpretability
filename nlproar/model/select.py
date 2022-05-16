
from .rnn_single_sequence_to_class import RNNSingleSequenceToClass
from .rnn_multiple_sequence_to_class import RNNMultipleSequenceToClass
from .roberta_single_sequence_to_class import RobertaSingleSequenceToClass
from .roberta_multiple_sequence_to_class import RobertaMultipleSequenceToClass
from .longformer_single_sequence_to_class import LongformerSingleSequenceToClass
from .longformer_multiple_sequence_to_class import LongformerMultipleSequenceToClass
from .xlnet_single_sequence_to_class import XLNetSingleSequenceToClass
from .xlnet_multiple_sequence_to_class import XLNetMultipleSequenceToClass

def select_single_sequence_to_class(model_type):
    return ({
        'rnn': RNNSingleSequenceToClass,
        'roberta': RobertaSingleSequenceToClass,
        'longformer': LongformerSingleSequenceToClass,
        'xlnet': XLNetSingleSequenceToClass
    })[model_type]

def select_multiple_sequence_to_class(model_type):
    return ({
        'rnn': RNNMultipleSequenceToClass,
        'roberta': RobertaMultipleSequenceToClass,
        'longformer': LongformerMultipleSequenceToClass,
        'xlnet': XLNetMultipleSequenceToClass
    })[model_type]
