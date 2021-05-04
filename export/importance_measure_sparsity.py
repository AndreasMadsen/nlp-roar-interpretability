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
    importance_normalized =importance / np.sum(importance)
    sorted_cumsum = np.cumsum(np.sort(importance_normalized))
    return (
        sorted_cumsum.size,
        sorted_cumsum.size - np.sum(sorted_cumsum <= 0.20),
        sorted_cumsum.size - np.sum(sorted_cumsum <= 0.10),
        sorted_cumsum.size - np.sum(sorted_cumsum <= 0.05)
    )

def _aggregate_importance(df):
    return pd.Series(
        _aggregate_importance_fast(df['importance'].to_numpy()),
        index=["length", "p80", "p90", "p95"])

def _compute_stats_and_format(col, percentage=False):
    mean = np.mean(col)
    sem = scipy.stats.sem(col)

    if sem < 1e-12:
        ci = 0
    else:
        ci = np.abs(scipy.stats.t.ppf(0.025, df=col.size, scale=sem))

    if percentage:
        return f"${mean:.2%} \\pm {ci:.2%}$"
    else:
        return f"${mean:.2f} \\pm {ci:.2f}$"

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
    importance_measure_mapping = {
        'r': 'Random',
        'a': 'Attention',
        'g': 'Gradient',
        'i': 'Integrated Gradient'
    }

    if args.stage in ['both', 'preprocess']:
        # Read CSV files into a dataframe and progressively aggregate the data
        df_partials = []
        df_partials_keys = []
        for file in tqdm(sorted(glob.glob(f'{args.persistent_dir}/results/importance_measure/*.csv.gz')),
                         desc='Parsing and summarzing CSVs'):
            filename = path.basename(file)
            dataset, seed, measure = re.match(r'([0-9A-Za-z-]+)_s-(\d+)_m-([a-z])', filename).groups()

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
            df_partial = pd.melt(df_partial,
                                id_vars=['length'],
                                value_vars=['p80', 'p90', 'p95'],
                                var_name='percentage',
                                value_name='absolute',
                                ignore_index=False)
            df_partial['relative'] = df_partial['absolute'] / df_partial['length']
            df_partials.append(df_partial)
            df_partials_keys.append((dataset[:-4], int(seed), importance_measure_mapping[measure]))

        df = pd.concat(df_partials, keys=df_partials_keys, names=['dataset', 'seed', 'importance_measure'])

    if args.stage in ['preprocess']:
        os.makedirs(f'{args.persistent_dir}/pandas', exist_ok=True)
        df.to_pickle(f'{args.persistent_dir}/pandas/sparsity.pd.pkl.xz')
    if args.stage in ['plot']:
        df = pd.read_pickle(f'{args.persistent_dir}/pandas/sparsity.pd.pkl.xz')

    if args.stage in ['both', 'plot']:
        print(df)

        # Average over seeds
        latex_df = (
            df.loc[pd.IndexSlice[:, :, :, 'train', :]]
                .groupby(["dataset", "seed", "importance_measure", "percentage"])
                .agg({ 'length': 'mean', 'absolute': 'mean', 'relative': 'mean' })
                .groupby(["dataset", "importance_measure", "percentage"])
                .agg({
                    'length': partial(_compute_stats_and_format, percentage=False),
                    'absolute': partial(_compute_stats_and_format, percentage=False),
                    'relative': partial(_compute_stats_and_format, percentage=True)
                })
                .reset_index()
                .merge(dataset_mapping, on="dataset")
                .pivot(
                    index=['dataset_pretty', 'importance_measure', 'length'],
                    columns=['percentage'],
                    values=['absolute', 'relative']
                )
                .reset_index(level='length')
        )
        print(latex_df)
        os.makedirs(f"{args.persistent_dir}/tables", exist_ok=True)
        latex_df.to_latex(f'{args.persistent_dir}/tables/attention_sparsity.tex', escape=False, index_names=False, multirow=True)

        # Draw figure
        plot_df = (
            df.loc[pd.IndexSlice[:, :, :, 'train', :]]
                .reset_index()
                .merge(dataset_mapping, on="dataset")
        )

        p = (p9.ggplot(plot_df, p9.aes(x='absolute', fill='percentage'))
            + p9.geom_histogram(p9.aes(y=p9.after_stat('density')), position = "identity", alpha=0.5, bins=50)
            + p9.facet_grid('dataset_pretty ~ importance_measure', scales='free')
            + p9.labs(x = 'number of tokens attended to'))
        # Save plot, the 6.30045 is the \linewidth of \begin{figure*} in the LaTeX document
        os.makedirs(f'{args.persistent_dir}/plots', exist_ok=True)
        p.save(f'{args.persistent_dir}/plots/importance_measure_sparsity_y-c.pdf', width=6.30045, height=7, units='in')
        p.save(f'{args.persistent_dir}/plots/importance_measure_sparsity_y-c.png', width=6.30045, height=7, units='in')

        p = (p9.ggplot(plot_df, p9.aes(x='relative', fill='percentage'))
            + p9.geom_histogram(p9.aes(y=p9.after_stat('density')), position = "identity", alpha=0.5, bins=50)
            + p9.facet_grid('dataset_pretty ~ importance_measure')
            + p9.labs(x = 'relative number of tokens attended to'))
        # Save plot, the 6.30045 is the \linewidth of \begin{figure*} in the LaTeX document
        os.makedirs(f'{args.persistent_dir}/plots', exist_ok=True)
        p.save(f'{args.persistent_dir}/plots/importance_measure_sparsity_y-q.pdf', width=6.30045, height=7, units='in')
        p.save(f'{args.persistent_dir}/plots/importance_measure_sparsity_y-q.png', width=6.30045, height=7, units='in')
