import argparse
import os.path as path

from comp550.dataset import SNLIDataModule, StanfordSentimentDataset

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

    print('SST ...')
    sst = StanfordSentimentDataset(cachedir=f'{args.persistent_dir}/cache')
    sst.prepare_data()

    print('SNLI ...')
    snli = SNLIDataModule(cachedir=f'{args.persistent_dir}/cache')
    snli.prepare_data()

    print("Download complete!")
