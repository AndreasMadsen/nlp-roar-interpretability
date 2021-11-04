import argparse
import os
import os.path as path
import json
import glob
import re
from functools import partial

import numba
import scipy.stats
import numpy as np
import pandas as pd
import plotnine as p9
from tqdm import tqdm

from comp550.util import generate_experiment_id

thisdir = os.path.dirname(os.path.realpath(__file__))
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

def ratio_confint(df):
    """Implementes a ratio-confidence interval

    The idea is to project to logits space, then assume a normal distribution,
    and then project back to the inital space.

    Method proposed here: https://stats.stackexchange.com/questions/263516
    """
    x = df['mass']
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
        'n': len(x)
    })

def _read_csv_tqdm(file, dtype=None, desc=None, leave=True):
    return pd.concat(tqdm(pd.read_csv(file, dtype=dtype, chunksize=1_000_000, usecols=list(dtype.keys())),
                     desc=desc, leave=leave))

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
    importance_measure_mapping = pd.DataFrame([
        {'importance_measure': 'a', 'importance_measure_pretty': 'Attention'},
        {'importance_measure': 'g', 'importance_measure_pretty': 'Gradient'},
        {'importance_measure': 'i', 'importance_measure_pretty': 'Integrated Gradient'},
        {'importance_measure': 'r', 'importance_measure_pretty': 'Random'}
    ])

    if args.stage in ['both', 'preprocess']:
        # Read CSV files into a dataframe and progressively aggregate the data
        df_partials = []
        df_partials_keys = []
        for file in tqdm(sorted(glob.glob(f'{args.persistent_dir}/results/importance_measure/*.csv.gz')),
                         desc='Parsing and summarzing CSVs'):
            filename = path.basename(file)

            dataset, seed, measure, riemann_samples = re.match(r'([0-9A-Za-z-]+)_s-(\d+)_m-([a-z])_rs-(\d+)', filename).groups()
            if (measure == 'i' and riemann_samples != '50') or measure == 'm':
                continue

            df_partial = _read_csv_tqdm(file, desc=f'Reading {filename}', leave=False, dtype={
                'split': pd.CategoricalDtype(categories=["train", "val", "test"], ordered=True),
                'observation': np.int32,
                'index': np.int32,
                'token': np.int32,
                'importance': np.float32,
                'walltime': np.float32
            })

            tqdm.pandas(desc=f"Aggregating {filename}", leave=False)
            df_partial = df_partial.groupby(['split', 'observation'], observed=True).progress_apply(_aggregate_importance)
            df_partials.append(df_partial)
            df_partials_keys.append((dataset[:-4], int(seed), measure))

        df = pd.concat(df_partials, keys=df_partials_keys, names=['dataset', 'seed', 'importance_measure'])

    if args.stage in ['preprocess']:
        os.makedirs(f'{args.persistent_dir}/pandas', exist_ok=True)
        df.to_pickle(f'{args.persistent_dir}/pandas/sparsity.pd.pkl.xz')
    if args.stage in ['plot']:
        df = pd.read_pickle(f'{args.persistent_dir}/pandas/sparsity.pd.pkl.xz')

    if args.stage in ['both', 'plot']:
        df = (
            df.loc[pd.IndexSlice[:, :, :, 'train', :]]
                .melt(value_vars=["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10",
                                "q10", "q20", "q30", "q40", "q50", "q60", "q70", "q80", "q90"],
                    value_name="mass",
                    var_name="k",
                    ignore_index=False)
                .assign(
                    strategy=lambda x: np.where(x['k'].str.startswith('c'), 'count', 'quantile'),
                    k=lambda x: pd.to_numeric(x['k'].str.slice(1))
                )
                .groupby(["dataset", "seed", "importance_measure", "strategy", "k"])
                .agg({ 'mass': 'mean' })
                .reset_index()
                .groupby(["dataset", "importance_measure", "strategy", "k"])
                .apply(ratio_confint)
                .reset_index()
                .merge(dataset_mapping, on='dataset')
                .drop(['dataset'], axis=1)
                .merge(importance_measure_mapping, on='importance_measure')
                .drop(['importance_measure'], axis=1)
                .set_index(['dataset_pretty', 'strategy', 'k', 'importance_measure_pretty'])
        )

        # Generate result table
        for strategy in ['count', 'quantile']:
            experiment_id = generate_experiment_id('sparsity', strategy=strategy)

            # Generate plot
            p = (p9.ggplot(df.loc[pd.IndexSlice[:, strategy, :, :]].reset_index(), p9.aes(x='k'))
                + p9.geom_ribbon(p9.aes(ymin='lower', ymax='upper', fill='importance_measure_pretty'), alpha=0.35)
                + p9.geom_line(p9.aes(y='mean', color='importance_measure_pretty'))
                + p9.geom_point(p9.aes(y='mean', color='importance_measure_pretty', shape='importance_measure_pretty'))
                + p9.facet_grid('dataset_pretty ~ .', scales='free_x')
                + p9.labs(y='', color='', shape='')
                + p9.scale_y_continuous(labels = lambda ticks: [f'{tick:.0%}' for tick in ticks])
                + p9.scale_color_manual(
                    values = ['#F8766D', '#A3A500', '#00BF7D', '#00B0F6'],
                    breaks = ['Attention', 'Gradient', 'Integrated Gradient', 'Random']
                )
                + p9.scale_shape_manual(
                    values = ['o', '^', 's', 'v'],
                    breaks = ['Attention', 'Gradient', 'Integrated Gradient', 'Random']
                )
                + p9.guides(fill=False, color = p9.guide_legend(nrow = 2))
                + p9.theme(plot_margin=0,
                        legend_box = "vertical", legend_position="bottom",
                        text=p9.element_text(size=12))
            )

            if strategy == 'count':
                p += p9.scale_x_continuous(name='nb. tokens', breaks=range(0, 11, 2))
            elif strategy == 'quantile':
                p += p9.scale_x_continuous(name='% tokens', breaks=range(0, 91, 20))

            p.save(f'{args.persistent_dir}/plots/{experiment_id}.pdf', width=3.03209, height=7, units='in')
            p.save(f'{args.persistent_dir}/plots/{experiment_id}.png', width=3.03209, height=7, units='in')

        df_latex = (
            df
                .loc[pd.IndexSlice[:, 'count', [1, 10], ['Attention', 'Gradient', 'Integrated Gradient']]]
                .droplevel(['strategy'])
                .assign(**{
                    'total importance': lambda x: [
                        f'${mean:.0%}^{{+{upper-mean:.1%}}}_{{-{mean-lower:.1%}}}$'.replace('%', '\\%')
                        for mean, lower, upper
                        in zip(x['mean'], x['lower'], x['upper'])
                    ]
                })
                .drop(['mean', 'lower', 'upper', 'n'], axis=1)
                .reset_index()
                .rename(columns={
                    'dataset_pretty': 'dataset',
                    'importance_measure_pretty': 'importance measure'
                })
                .pivot(
                    index=['dataset', 'importance measure'],
                    columns=['k']
                )
        )
        os.makedirs(f"{args.persistent_dir}/tables", exist_ok=True)
        df_latex.to_latex(f'{args.persistent_dir}/tables/sparsity.tex', escape=False, multirow=True)
