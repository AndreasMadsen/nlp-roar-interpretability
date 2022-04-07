import argparse
import os.path as path

from nlproar.dataset import MimicDataset, SNLIDataset, SSTDataset, IMDBDataset, BabiDataset
from nlproar.model import RNNSingleSequenceToClass, RobertaSingleSequenceToClass

thisdir = path.dirname(path.realpath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--persistent-dir',
                    action='store',
                    default=path.realpath(path.join(thisdir, '..')),
                    type=str,
                    help='Directory where all persistent data will be stored')

if __name__ == "__main__":
    print('Starting ...')
    args = parser.parse_args()

    # download and prepear datasets
    for model_type in ['rnn', 'roberta']:
        print('Mimic ...')
        mimic = MimicDataset(cachedir=f'{args.persistent_dir}/cache', mimicdir=f'{args.persistent_dir}/mimic', model_type=model_type)
        mimic.prepare_data()

        print('SST ...')
        sst = SSTDataset(cachedir=f'{args.persistent_dir}/cache', model_type=model_type)
        sst.prepare_data()

        print('SNLI ...')
        snli = SNLIDataset(cachedir=f'{args.persistent_dir}/cache', model_type=model_type)
        snli.prepare_data()

        print('IMDB ...')
        imdb = IMDBDataset(cachedir=f'{args.persistent_dir}/cache', model_type=model_type)
        imdb.prepare_data()

        print('Babi ...')
        for i in range(1, 4):
            babi = BabiDataset(cachedir=f'{args.persistent_dir}/cache', model_type=model_type, task=i)
            babi.prepare_data()

    # download models
    rnn = RNNSingleSequenceToClass(f'{args.persistent_dir}/cache', sst.embedding())
    roberta = RobertaSingleSequenceToClass(f'{args.persistent_dir}/cache')

    print("Download complete!")
