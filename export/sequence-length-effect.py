
import glob
import json
import argparse
import os
import os.path as path
from functools import partial

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import scipy
import plotnine as p9
from scipy.stats import bootstrap
import torchmetrics

from nlproar.dataset import SNLIDataset, SSTDataset, IMDBDataset, BabiDataset, MimicDataset
from nlproar.util import generate_experiment_id

def ratio_confint(partial_df):
    """Implementes a ratio-confidence interval

    The idea is to project to logits space, then assume a normal distribution,
    and then project back to the inital space.

    Method proposed here: https://stats.stackexchange.com/questions/263516
    """
    column_name = partial_df.loc[:, 'test_metric'].iat[0]
    x = partial_df.loc[:, column_name].to_numpy()
    mean = np.mean(x)

    if np.all(x[0] == x):
        lower = mean
        upper = mean
    else:
        res = bootstrap((x, ), np.mean, confidence_level=0.95, random_state=np.random.default_rng(0))
        lower = res.confidence_interval.low
        upper = res.confidence_interval.high

    return pd.Series({
        'lower': lower,
        'mean': mean,
        'upper': upper,
        'format': f'${mean:.0%}^{{+{upper-mean:.1%}}}_{{-{mean-lower:.1%}}}$'.replace('%', '\\%'),
        'n': len(x)
    })

def dataset_stats(Loader, Metric, cachedir):
    dataset = Loader(cachedir=cachedir, model_type='rnn', num_workers=0)
    dataset.prepare_data()
    dataset.setup('fit')
    dataset.setup('test')

    labels = np.zeros(len(dataset.label_names))
    for batch in tqdm(dataset.train_dataloader(), desc=f'Counting labels', leave=False):
        for label in batch.label:
            labels[label] += 1
    majority_class = np.argmax(labels)

    metric = Metric(num_classes=len(dataset.label_names), compute_on_step=False)
    for batch in tqdm(dataset.test_dataloader(), desc=f'Measuring majority classifier', leave=False):
        predict_labels = torch.full((len(batch.label), ), majority_class)
        metric.update(predict_labels, batch.label)

    return pd.Series({
        'dataset': dataset.name,
        'majority': metric.compute().numpy()
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
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    args, unknown = parser.parse_known_args()

    datasets = {
        'sst': (SSTDataset, partial(torchmetrics.F1, average='macro')),
        'snli': (SNLIDataset, partial(torchmetrics.F1, average='micro')),
        'imdb': (IMDBDataset, partial(torchmetrics.F1, average='macro')),
        'babi-1': (partial(BabiDataset, task=1), torchmetrics.Accuracy),
        'babi-2': (partial(BabiDataset, task=2), torchmetrics.Accuracy),
        'babi-3': (partial(BabiDataset, task=3), torchmetrics.Accuracy),
        'mimic-d': (partial(MimicDataset, subset='diabetes', mimicdir=f'{args.persistent_dir}/mimic'),
                    partial(torchmetrics.F1, average='macro')),
        'mimic-a': (partial(MimicDataset, subset='anemia', mimicdir=f'{args.persistent_dir}/mimic'),
                    partial(torchmetrics.F1, average='macro')),
    }

    dataset_mapping = pd.DataFrame([
        {'dataset': 'sst', 'dataset_pretty': 'SST', 'test_metric': 'f1_test'},
        {'dataset': 'snli', 'dataset_pretty': 'SNLI', 'test_metric': 'f1_test'},
        {'dataset': 'imdb', 'dataset_pretty': 'IMDB', 'test_metric': 'f1_test'},
        {'dataset': 'mimic-a', 'dataset_pretty': 'Anemia', 'test_metric': 'f1_test'},
        {'dataset': 'mimic-d', 'dataset_pretty': 'Diabetes', 'test_metric': 'f1_test'},
        {'dataset': 'babi-1', 'dataset_pretty': 'bAbI-1', 'test_metric': 'acc_test'},
        {'dataset': 'babi-2', 'dataset_pretty': 'bAbI-2', 'test_metric': 'acc_test'},
        {'dataset': 'babi-3', 'dataset_pretty': 'bAbI-3', 'test_metric': 'acc_test'},
    ])

    model_mapping = pd.DataFrame([
        {'model_type': 'rnn', 'model_type_pretty': 'BiLSTM-Attention'},
        {'model_type': 'roberta', 'model_type_pretty': 'RoBERTa'}
    ])

    if args.stage in ['both', 'preprocess']:
        # Read JSON files into dataframe
        results = []
        for file in tqdm(glob.glob(f'{args.persistent_dir}/results/roar/*.json'), desc='Loading .json files'):
            with open(file, 'r') as fp:
                try:
                    results.append(json.load(fp))
                except json.decoder.JSONDecodeError:
                    print(f'{file} has a format error')
        model_df = pd.DataFrame(results)
        model_df = model_df.loc[(model_df['k'] == 100) & (model_df['strategy'] == 'quantile')]

        # Summarize each dataset
        dataset_summaries = []
        for dataset_loader, dataset_metric in tqdm(datasets.values(), desc='Summarizing datasets'):
            dataset_summaries.append(dataset_stats(dataset_loader, dataset_metric, cachedir=args.persistent_dir + '/cache'))
        dataset_df = pd.DataFrame(dataset_summaries)

        # Compute confint and mean for each group
        df = (model_df
            .merge(dataset_mapping, on='dataset')
            .merge(model_mapping, on='model_type')
            .drop(['model_type'], axis=1)
            .groupby([
                'dataset', 'dataset_pretty', 'model_type_pretty'
            ])
            .apply(ratio_confint)
            .reset_index()
            .merge(dataset_df, on='dataset')
            .drop(['dataset'], axis=1)
        )

    if args.stage in ['preprocess']:
        os.makedirs(f'{args.persistent_dir}/pandas', exist_ok=True)
        df.to_pickle(f'{args.persistent_dir}/pandas/sequence-length.pd.pkl.xz')
    if args.stage in ['plot']:
        df = pd.read_pickle(f'{args.persistent_dir}/pandas/sequence-length.pd.pkl.xz')

    if args.stage in ['both', 'plot']:
        df = df.reset_index()
        df = df.rename(columns={
            'dataset_pretty': 'Dataset',
            'model_type_pretty': 'Model',
            'format': 'Performance',
        })
        df = df.assign(
            majority = lambda df: df['majority'].map(lambda v: f'${v*100:.0f}\\%$'),
        )
        df = df.pivot(
            index=['Dataset', 'majority'],
            columns='Model',
            values='Performance'
        )
        df = df.style.to_latex()
        print(df)
