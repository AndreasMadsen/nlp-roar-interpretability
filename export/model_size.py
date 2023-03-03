import argparse
from functools import partial
import os.path as path
import os

from tqdm import tqdm

from nlproar.dataset import SNLIDataset, SSTDataset, IMDBDataset, BabiDataset, MimicDataset
from nlproar.model import select_multiple_sequence_to_class, select_single_sequence_to_class

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Print model size / number of parameters for BiLSTM-attention")
parser.add_argument("--persistent-dir",
                    action="store",
                    default=os.path.realpath(os.path.join(thisdir, "..")),
                    type=str,
                    help="Directory where all persistent data will be stored")

if __name__ == "__main__":
    args = parser.parse_args()

    # Get dataset class and model class
    SingleSequenceToClass = select_single_sequence_to_class('rnn')
    MultipleSequenceToClass = select_multiple_sequence_to_class('rnn')

    datasets = {
        'sst': (SSTDataset, SingleSequenceToClass),
        'snli': (SNLIDataset, MultipleSequenceToClass),
        'imdb': (IMDBDataset, SingleSequenceToClass),
        'babi-1': (partial(BabiDataset, task=1), partial(MultipleSequenceToClass, hidden_size=32)),
        'babi-2': (partial(BabiDataset, task=2), partial(MultipleSequenceToClass, hidden_size=32)),
        'babi-3': (partial(BabiDataset, task=3), partial(MultipleSequenceToClass, hidden_size=32)),
        'mimic-d': (partial(MimicDataset, subset='diabetes', mimicdir=f'{args.persistent_dir}/mimic'), SingleSequenceToClass),
        'mimic-a': (partial(MimicDataset, subset='anemia', mimicdir=f'{args.persistent_dir}/mimic'), SingleSequenceToClass)
    }

    for dataset, (dataset_cls, model_cls) in tqdm(datasets.items()):
        # Load dataset
        dataset = dataset_cls(cachedir=f"{args.persistent_dir}/cache", model_type='rnn', num_workers=0)
        dataset.prepare_data()

        # Load model
        model = model_cls(cachedir=f"{args.persistent_dir}/cache",
                          embedding=dataset.embedding(),
                          num_of_classes=len(dataset.label_names))
        num_of_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        tqdm.write(f'{dataset.name}: {num_of_parameters}')
