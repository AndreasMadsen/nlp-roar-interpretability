
import glob
import json
import argparse
import os
import os.path as path
from functools import partial

import pandas as pd
import numpy as np
import scipy
import plotnine as p9

import os.path as path
from comp550.dataset import StanfordSentimentDataset, SNLIDataModule, IMDBDataModule, MimicDataset, BabiDataModule

def ratio_confint(partial_df):
    """Implementes a ratio-confidence interval

    The idea is to project to logits space, then assume a normal distribution,
    and then project back to the inital space.

    Method proposed here: https://stats.stackexchange.com/questions/263516
    """
    column_name = partial_df.loc[:, 'test_metric'].iat[0]
    x = partial_df[column_name]
    logits = scipy.special.logit(x)
    mean = np.mean(logits)
    sem = scipy.stats.sem(logits)
    lower, upper = scipy.stats.t.interval(0.95, len(x) - 1,
                                          loc=mean,
                                          scale=sem)

    lower = scipy.special.expit(lower)
    mean = scipy.special.expit(mean)
    upper = scipy.special.expit(upper)
    if np.isnan(sem):
        lower, upper = mean, mean

    return pd.Series({
        'lower': lower,
        'mean': mean,
        'upper': upper,
        'format': f'${mean:.0%}^{{+{upper-mean:.1%}}}_{{-{mean-lower:.1%}}}$'.replace('%', '\\%'),
        'n': len(x)
    })

def dataset_stats(partial_df, cachedir='./'):
    Loader = partial_df.loc[:, 'loader'].iat[0]
    dataset = Loader(cachedir=cachedir, num_workers=0)
    dataset.prepare_data()
    dataset.setup('fit')
    dataset.setup('test')

    summaries = {}
    dataloaders = [
        ('train', dataset.train_dataloader()),
        ('val', dataset.val_dataloader()),
        ('test', dataset.test_dataloader())
    ]
    for split_name, split_iter in dataloaders:
        lengths = []
        observations = [0] * len(dataset.label_names)
        for batch in split_iter:
            lengths += list(batch['length'])
            for value, count in zip(*np.unique(batch['label'], return_counts=True)):
                observations[value] += count

        # Hack for bAbI-x
        if len(observations) > 3:
            observations = [sum(observations)]

        summaries[split_name] = {
            'length': np.mean(lengths),
            'count': sum(observations),
            'observations': observations
        }

    return pd.Series({
       'vocab_size': len(dataset.vocabulary),
       'avg_length': np.average(
           [summary['length'] for summary in summaries.values()],
           weights=[summary['count'] for summary in summaries.values()]),
       'train_size': '/'.join(map(str, summaries['train']['observations'])),
       'val_size': '/'.join(map(str, summaries['val']['observations'])),
       'test_size': '/'.join(map(str, summaries['test']['observations']))
    })

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

    dataset_mapping = pd.DataFrame([
        {'dataset': 'sst', 'print_name': 'SST', 'test_metric': 'f1_test',
         'loader': StanfordSentimentDataset},
        {'dataset': 'snli', 'print_name': 'SNLI', 'test_metric': 'f1_test',
         'loader': SNLIDataModule},
        {'dataset': 'imdb', 'print_name': 'IMDB', 'test_metric': 'f1_test',
         'loader': IMDBDataModule},
        {'dataset': 'mimic-anemia', 'print_name': 'Anemia', 'test_metric': 'f1_test',
         'loader': partial(MimicDataset, mimicdir=args.persistent_dir + '/mimic', subset='anemia')},
        {'dataset': 'mimic-diabetes', 'print_name': 'Diabetes', 'test_metric': 'f1_test',
         'loader': partial(MimicDataset, mimicdir=args.persistent_dir + '/mimic', subset='diabetes')},
        {'dataset': 'babi_t-1', 'print_name': 'bAbI-1', 'test_metric': 'acc_test',
         'loader': partial(BabiDataModule, task_idx=1)},
        {'dataset': 'babi_t-2', 'print_name': 'bAbI-2', 'test_metric': 'acc_test',
         'loader': partial(BabiDataModule, task_idx=2)},
        {'dataset': 'babi_t-3', 'print_name': 'bAbI-3', 'test_metric': 'acc_test',
         'loader': partial(BabiDataModule, task_idx=3)},
    ])

    # Read JSON files into dataframe
    results = []
    for file in glob.glob(f'{args.persistent_dir}/results/*.json'):
        with open(file, 'r') as fp:
            try:
                results.append(json.load(fp))
            except json.decoder.JSONDecodeError:
                print(f'{file} has a format error')
    df = pd.DataFrame(results)
    df = df.loc[df['roar'] == False]
    df = df.merge(dataset_mapping, on='dataset')

    performance_df = df.groupby(['print_name'], sort=False).apply(ratio_confint)
    dataset_df = df.groupby(['print_name'], sort=False).apply(
        partial(dataset_stats, cachedir=args.persistent_dir + '/cache')
    )


    summary_df = performance_df.merge(dataset_df, on='print_name')
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    print(summary_df)
