
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
from scipy.stats import bootstrap

from nlproar.dataset import SNLIDataset, SSTDataset, IMDBDataset, BabiDataset, MimicDataset

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

def dataset_stats(Loader, cachedir):
    dataset = Loader(cachedir=cachedir, model_type='rnn', num_workers=0)
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
        for batch in tqdm(split_iter, desc=f'Summarizing {split_name} split', leave=False):
            lengths += batch.length.tolist()

        summaries[split_name] = {
            'length': np.mean(lengths),
            'count': len(lengths),
        }

    return pd.Series({
        'dataset': dataset.name,
        'vocab_size': len(dataset.vocabulary),
        'train_size': summaries['train']['count'],
        'valid_size': summaries['val']['count'],
        'test_size': summaries['test']['count'],
        'avg_length': np.average(
            [summary['length'] for summary in summaries.values()],
            weights=[summary['count'] for summary in summaries.values()]
        )
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
        {'dataset': 'sst', 'dataset_pretty': 'SST', 'test_metric': 'f1_test', 'reference': '$81\\%$'},
        {'dataset': 'snli', 'dataset_pretty': 'SNLI', 'test_metric': 'f1_test', 'reference': '$88\\%$'},
        {'dataset': 'imdb', 'dataset_pretty': 'IMDB', 'test_metric': 'f1_test', 'reference': '$78\\%$'},
        {'dataset': 'mimic-a', 'dataset_pretty': 'Anemia', 'test_metric': 'f1_test', 'reference': '$92\\%$'},
        {'dataset': 'mimic-d', 'dataset_pretty': 'Diabetes', 'test_metric': 'f1_test', 'reference': '$79\\%$'},
        {'dataset': 'babi-1', 'dataset_pretty': 'bAbI-1', 'test_metric': 'acc_test', 'reference': '$100\\%$'},
        {'dataset': 'babi-2', 'dataset_pretty': 'bAbI-2', 'test_metric': 'acc_test', 'reference': '$48\\%$'},
        {'dataset': 'babi-3', 'dataset_pretty': 'bAbI-3', 'test_metric': 'acc_test', 'reference': '$62\\%$'}
    ])

    model_mapping = pd.DataFrame([
        {'model_type': 'rnn', 'model_type_pretty': 'BiLSTM-Attention'},
        {'model_type': 'roberta', 'model_type_pretty': 'RoBERTa'}
    ])

    datasets = {
        'sst': SSTDataset,
        'snli': SNLIDataset,
        'imdb': IMDBDataset,
        'babi-1': partial(BabiDataset, task=1),
        'babi-2': partial(BabiDataset, task=2),
        'babi-3': partial(BabiDataset, task=3),
        'mimic-d': partial(MimicDataset, subset='diabetes', mimicdir=f'{args.persistent_dir}/mimic'),
        'mimic-a': partial(MimicDataset, subset='anemia', mimicdir=f'{args.persistent_dir}/mimic'),
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
        for dataset_loader in tqdm(datasets.values(), desc='Summarizing datasets'):
            summaries.append(dataset_stats(dataset_loader, cachedir=args.persistent_dir + '/cache'))
        summaries_df = pd.DataFrame(summaries)

        df = (results_df
            .merge(dataset_mapping, on='dataset')
            .groupby(['dataset', 'dataset_pretty', 'reference', 'model_type'])
            .apply(ratio_confint)
            .reset_index()
            .merge(summaries_df, on='dataset')
            .merge(model_mapping, on='model_type')
            .drop(['lower', 'upper', 'n', 'mean', 'dataset', 'model_type'], axis=1)
        )

    if args.stage in ['preprocess']:
        os.makedirs(f'{args.persistent_dir}/pandas', exist_ok=True)
        df.to_pickle(f'{args.persistent_dir}/pandas/dataset.pd.pkl.xz')
    if args.stage in ['plot']:
        df = pd.read_pickle(f'{args.persistent_dir}/pandas/dataset.pd.pkl.xz')

    if args.stage in ['both', 'plot']:
        print(df)

        print(df
            .reset_index()
            .rename(columns={
                'dataset_pretty': 'Dataset',
                'format': 'Faithfulness'
            })
            .pivot(
                index=['Dataset'],
                columns='model_type_pretty',
                values='Faithfulness'
            )
            .style.to_latex()
        )

        print(df
            .reset_index()
            .rename(columns={
                'dataset_pretty': 'Dataset',
                'format': 'Faithfulness'
            })
            .pivot(
                index=['Dataset', 'train_size', 'valid_size', 'test_size', 'reference'],
                columns='model_type_pretty',
                values='Faithfulness'
            )
            .style.to_latex()
        )