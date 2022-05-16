import argparse
import os
import os.path as path
import json
import glob
import re
import gzip
from functools import partial
from multiprocessing import Pool

import numba
import scipy.stats
import numpy as np
import pandas as pd
import plotnine as p9
from tqdm import tqdm
from scipy.stats import bootstrap

from nlproar.util import generate_experiment_id

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--persistent-dir',
                    action='store',
                    default=path.realpath(path.join(thisdir, '..')),
                    type=str,
                    help='Directory where all persistent data will be stored')
parser.add_argument('--num-workers',
                    action='store',
                    default=4,
                    type=int,
                    help='The number of workers to use in data loading')
parser.add_argument('--stage',
                    action='store',
                    default='both',
                    type=str,
                    choices=['preprocess', 'plot', 'both'],
                    help='Which export stage should be performed. Mostly just useful for debugging.')

@numba.jit(nopython=True)
def _aggregate_importance_fast(importance):
    importance_normalized = importance / np.sum(importance)
    sorted_cumsum = np.cumsum(np.sort(importance_normalized)[::-1])
    return (
        sorted_cumsum.size,

        sorted_cumsum[0] if 0 < sorted_cumsum.size else 1.0,
        sorted_cumsum[1] if 1 < sorted_cumsum.size else 1.0,
        sorted_cumsum[2] if 2 < sorted_cumsum.size else 1.0,
        sorted_cumsum[3] if 3 < sorted_cumsum.size else 1.0,
        sorted_cumsum[4] if 4 < sorted_cumsum.size else 1.0,
        sorted_cumsum[5] if 5 < sorted_cumsum.size else 1.0,
        sorted_cumsum[6] if 6 < sorted_cumsum.size else 1.0,
        sorted_cumsum[7] if 7 < sorted_cumsum.size else 1.0,
        sorted_cumsum[8] if 8 < sorted_cumsum.size else 1.0,
        sorted_cumsum[9] if 9 < sorted_cumsum.size else 1.0,

        sorted_cumsum[int(sorted_cumsum.size * 0.10)],
        sorted_cumsum[int(sorted_cumsum.size * 0.20)],
        sorted_cumsum[int(sorted_cumsum.size * 0.30)],
        sorted_cumsum[int(sorted_cumsum.size * 0.40)],
        sorted_cumsum[int(sorted_cumsum.size * 0.50)],
        sorted_cumsum[int(sorted_cumsum.size * 0.60)],
        sorted_cumsum[int(sorted_cumsum.size * 0.70)],
        sorted_cumsum[int(sorted_cumsum.size * 0.80)],
        sorted_cumsum[int(sorted_cumsum.size * 0.90)]
    )

def _aggregate_importance(df):
    return pd.Series(
        _aggregate_importance_fast(df['importance'].to_numpy()),
        index=["length",
               "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", 'c10',
               "q10", "q20", "q30", "q40", "q50", "q60", "q70", "q80", "q90"])


def ratio_confint(partial_df):
    """Implementes a ratio-confidence interval

    The idea is to project to logits space, then assume a normal distribution,
    and then project back to the inital space.

    Method proposed here: https://stats.stackexchange.com/questions/263516
    """
    x = partial_df.loc[:, 'mass'].to_numpy()
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
        'n': len(x)
    })

def parse_files(files):
    out = []

    for file in sorted(files):
        filename = path.basename(file)
        dataset, model, seed, measure, riemann_samples = re.match(r'([0-9a-z-]+)_([a-z]+)-pre_s-(\d+)_m-([a-z])_rs-(\d+)', filename).groups()
        if (measure == 'i' and riemann_samples != '50') or measure == 'm':
            continue
        out.append((file, (dataset, model, int(seed), measure)))

    return out

def process_csv(args):
    file, key = args

    try:
        df_partial = pd.read_csv(file, usecols=['split', 'observation', 'importance'], dtype={
            'split': pd.CategoricalDtype(categories=["train", "val", "test"], ordered=True),
            'observation': np.int32,
            'importance': np.float32,
        })
    except gzip.BadGzipFile as e:
        print(f'Bad file: {file}', flush=True)
        raise e

    df_partial = df_partial \
            .loc[df_partial['split'] == 'val', :] \
            .groupby(['observation']) \
            .apply(_aggregate_importance) \
            .reset_index() \
            .drop(['observation'], axis=1) \
            .mean() \
            .to_frame().T \
            .reset_index() \
            .drop(['index'], axis=1)

    return (key, df_partial)

if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    args, unknown = parser.parse_known_args()

    dataset_mapping = pd.DataFrame([
        {"dataset": "sst", "dataset_pretty": "SST"},
        {"dataset": "snli", "dataset_pretty": "SNLI"},
        {"dataset": "imdb", "dataset_pretty": "IMDB"},
        {"dataset": "babi-1", "dataset_pretty": "bAbI-1"},
        {"dataset": "babi-2", "dataset_pretty": "bAbI-2"},
        {"dataset": "babi-3", "dataset_pretty": "bAbI-3"},
        {"dataset": "mimic-a", "dataset_pretty": "Anemia"},
        {"dataset": "mimic-d", "dataset_pretty": "Diabetes"},
    ])

    model_mapping = pd.DataFrame([
        {'model_type': 'rnn', 'model_type_pretty': 'BiLSTM-Attention'},
        {'model_type': 'roberta', 'model_type_pretty': 'RoBERTa'}
    ])

    importance_measure_mapping = pd.DataFrame([
        {'importance_measure': 'a', 'importance_measure_pretty': 'Attention'},
        {'importance_measure': 'g', 'importance_measure_pretty': 'Gradient'},
        {'importance_measure': 't', 'importance_measure_pretty': 'Input times Gradient'},
        {'importance_measure': 'i', 'importance_measure_pretty': 'Integrated Gradient'},
        {'importance_measure': 'r', 'importance_measure_pretty': 'Random'}
    ])

    if args.stage in ['both', 'preprocess']:
        # Read CSV files into a dataframe and progressively aggregate the data
        df_partials_keys = []
        df_partials = []

        with Pool(args.num_workers) as pool:
            files = parse_files(glob.glob(f'{args.persistent_dir}/results/importance_measure/*-pre*.csv.gz'))
            for key, df_partial in tqdm(pool.imap_unordered(process_csv, files),
                                        total=len(files), desc='Parsing and summarzing CSVs'):
                df_partials_keys.append(key)
                df_partials.append(df_partial)

        df = pd.concat(df_partials, keys=df_partials_keys, names=['dataset', 'model_type', 'seed', 'importance_measure']) \
            .reset_index() \
            .drop(['level_4'], axis=1)

    if args.stage in ['preprocess']:
        os.makedirs(f'{args.persistent_dir}/pandas', exist_ok=True)
        df.to_pickle(f'{args.persistent_dir}/pandas/sparsity.pd.pkl.xz')
    if args.stage in ['plot']:
        df = pd.read_pickle(f'{args.persistent_dir}/pandas/sparsity.pd.pkl.xz')

    if args.stage in ['both', 'plot']:
        df = (df.merge(dataset_mapping, on='dataset')
               .merge(model_mapping, on='model_type')
               .merge(importance_measure_mapping, on='importance_measure')
               .drop(['dataset', 'model_type', 'importance_measure'], axis=1)
               .melt(value_vars=["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10",
                                "q10", "q20", "q30", "q40", "q50", "q60", "q70", "q80", "q90"],
                    id_vars=['dataset_pretty', 'model_type_pretty', 'importance_measure_pretty', 'seed'],
                    value_name="mass",
                    var_name="k")
                .assign(
                    strategy=lambda x: np.where(x['k'].str.startswith('c'), 'count', 'quantile'),
                    k=lambda x: pd.to_numeric(x['k'].str.slice(1))
                )
                .groupby(["dataset_pretty", "model_type_pretty", "importance_measure_pretty", "strategy", "k"])
                .apply(ratio_confint)
        )

        # Generate result table
        for strategy in ['count', 'quantile']:
            experiment_id = generate_experiment_id('sparsity', strategy=strategy)

            # Generate plot
            p = (p9.ggplot(df.loc[pd.IndexSlice[:, :, :, strategy, :]].reset_index(), p9.aes(x='k'))
                + p9.geom_ribbon(p9.aes(ymin='lower', ymax='upper', fill='importance_measure_pretty'), alpha=0.35)
                + p9.geom_line(p9.aes(y='mean', color='importance_measure_pretty'))
                + p9.geom_point(p9.aes(y='mean', color='importance_measure_pretty', shape='importance_measure_pretty'))
                + p9.facet_grid('dataset_pretty ~ model_type_pretty', scales='free_y')
                + p9.labs(y='', color='', shape='')
                + p9.scale_color_manual(
                    values = ['#F8766D', '#A3A500', '#00BF7D', '#00B0F6', '#E76BF3'],
                    breaks = ['Attention', 'Gradient', 'Input times Gradient', 'Integrated Gradient', 'Random']
                )
                + p9.scale_fill_manual(
                    values = ['#F8766D', '#A3A500', '#00BF7D', '#00B0F6', '#E76BF3'],
                    breaks = ['Attention', 'Gradient', 'Input times Gradient', 'Integrated Gradient', 'Random']
                )
                + p9.scale_shape_manual(
                    values = ['o', '^', 's', 'D', 'v'],
                    breaks = ['Attention', 'Gradient', 'Input times Gradient', 'Integrated Gradient', 'Random']
                )
                + p9.guides(fill=False)
                + p9.theme(plot_margin=0,
                        legend_box = "vertical", legend_position="bottom",
                        text=p9.element_text(size=12))
            )

            if strategy == 'count':
                p += p9.scale_x_continuous(name='nb. tokens', breaks=range(0, 11, 2))
                p += p9.scale_y_continuous(limits=[0, None], labels = lambda ticks: [f'{tick:.0%}' for tick in ticks])
            elif strategy == 'quantile':
                p += p9.scale_x_continuous(name='% tokens', breaks=range(0, 91, 20))
                p += p9.scale_y_continuous(limits=[0, 1], labels = lambda ticks: [f'{tick:.0%}' for tick in ticks])

            p.save(f'{args.persistent_dir}/plots/{experiment_id}.pdf', width=6.30045 + 0.2, height=7, units='in')
            p.save(f'{args.persistent_dir}/plots/{experiment_id}.png', width=6.30045 + 0.2, height=7, units='in')
