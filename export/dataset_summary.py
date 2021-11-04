
import glob
import json
import argparse
import os
import os.path as path
from functools import partial

from tqdm import tqdm
import pandas as pd
import numpy as np
import scipy
import plotnine as p9

from nlproar.dataset import SNLIDataset, SSTDataset, IMDBDataset, BabiDataset, MimicDataset

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

def dataset_stats(Loader, cachedir, avg_length='by-class'):
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
        for batch in tqdm(split_iter, desc=f'Summarizing {split_name} split', leave=False):
            lengths += batch.length.tolist()
            for value, count in zip(*np.unique(batch.label, return_counts=True)):
                observations[value] += count

        if avg_length == 'in-total':
            if len(observations) > 3:
                observations = [sum(observations)]

        summaries[split_name] = {
            'length': np.mean(lengths),
            'count': sum(observations),
            'observations': observations
        }

    return pd.Series({
        'dataset': dataset.name,
        'vocab_size': len(dataset.vocabulary),
        'avg_length': np.average(
            [summary['length'] for summary in summaries.values()],
            weights=[summary['count'] for summary in summaries.values()]
        ),
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
parser.add_argument('--stage',
                    action='store',
                    default='both',
                    type=str,
                    choices=['preprocess', 'plot', 'both'],
                    help='Which export stage should be performed. Mostly just useful for debugging.')

if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    args, unknown = parser.parse_known_args()

    dataset_mapping = pd.DataFrame([
        {'dataset': 'sst', 'dataset_pretty': 'SST', 'test_metric': 'f1_test'},
        {'dataset': 'snli', 'dataset_pretty': 'SNLI', 'test_metric': 'f1_test'},
        {'dataset': 'imdb', 'dataset_pretty': 'IMDB', 'test_metric': 'f1_test'},
        {'dataset': 'mimic-a', 'dataset_pretty': 'Anemia', 'test_metric': 'f1_test'},
        {'dataset': 'mimic-d', 'dataset_pretty': 'Diabetes', 'test_metric': 'f1_test'},
        {'dataset': 'babi-1', 'dataset_pretty': 'bAbI-1', 'test_metric': 'acc_test'},
        {'dataset': 'babi-2', 'dataset_pretty': 'bAbI-2', 'test_metric': 'acc_test'},
        {'dataset': 'babi-3', 'dataset_pretty': 'bAbI-3', 'test_metric': 'acc_test'}
    ])

    datasets = {
        'sst': (SSTDataset, 'by-class'),
        'snli': (SNLIDataset, 'by-class'),
        'imdb': (IMDBDataset, 'by-class'),
        'babi-1': (partial(BabiDataset, task=1), 'in-total'),
        'babi-2': (partial(BabiDataset, task=2), 'in-total'),
        'babi-3': (partial(BabiDataset, task=3), 'in-total'),
        'mimic-d': (partial(MimicDataset, subset='diabetes', mimicdir=f'{args.persistent_dir}/mimic'), 'by-class'),
        'mimic-a': (partial(MimicDataset, subset='anemia', mimicdir=f'{args.persistent_dir}/mimic'), 'by-class')
    }

    if args.stage in ['both', 'preprocess']:
        # Read JSON files into dataframe
        results = []
        for file in tqdm(glob.glob(f'{args.persistent_dir}/results/roar/*_s-[0-9].json'),
                        desc='Loading .json files'):
            with open(file, 'r') as fp:
                try:
                    results.append(json.load(fp))
                except json.decoder.JSONDecodeError:
                    print(f'{file} has a format error')
        results_df = pd.DataFrame(results)

        # Summarize each dataset
        summaries = []
        for dataset_loader, avg_length in tqdm(datasets.values(), desc='Summarizing datasets'):
            summaries.append(
                dataset_stats(dataset_loader, cachedir=args.persistent_dir + '/cache', avg_length=avg_length)
            )
        summaries_df = pd.DataFrame(summaries)

        # Hack for bAbI-x
        df = (
            results_df
            .merge(dataset_mapping, on='dataset')
            .groupby(['dataset', 'dataset_pretty'], sort=False, as_index=False).apply(ratio_confint)
            .merge(summaries_df, on='dataset')
        )

    if args.stage in ['preprocess']:
        os.makedirs(f'{args.persistent_dir}/pandas', exist_ok=True)
        df.to_pickle(f'{args.persistent_dir}/pandas/dataset.pd.pkl.xz')
    if args.stage in ['plot']:
        df = pd.read_pickle(f'{args.persistent_dir}/pandas/dataset.pd.pkl.xz')

    if args.stage in ['both', 'plot']:
        print(df)
