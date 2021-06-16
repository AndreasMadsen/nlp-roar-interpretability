import argparse
import os
import os.path as path
import json
import glob
import re
from functools import partial

import scipy.stats
import numpy as np
import pandas as pd
import plotnine as p9
from tqdm import tqdm

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument("--persistent-dir",
                    action="store",
                    default=os.path.realpath(os.path.join(thisdir, "..")),
                    type=str,
                    help="Directory where all persistent data will be stored")
parser.add_argument('--stage',
                    action='store',
                    default='both',
                    type=str,
                    choices=['preprocess', 'plot', 'both'],
                    help='Which export stage should be performed. Mostly just useful for debugging.')

def _precompute_rank(df):
    return pd.DataFrame({
        'split': df['split'],
        'observation': df['observation'],
        'index': df['index'],
        'token_gold': df['token'],
        'importance_gold': df['importance'],
        'importance_rank_gold':  np.argsort(df['importance'].to_numpy())[::-1]
    })

def _compute_stats_and_format(col, percentage=False):
    mean = np.mean(col)
    ci = np.abs(scipy.stats.t.ppf(0.025, df=col.size, scale=scipy.stats.sem(col)))

    if percentage:
        return f"${mean:.2%} \\pm {ci:.2%}$".replace('%', '\\%')
    else:
        return f"${mean:.2f} \\pm {ci:.2f}$"

def _aggregate_importance(df):
    abs_error = np.abs(df["importance"] - df["importance_gold"])
    rank_error = np.argsort(df["importance"].to_numpy())[::-1] != df["importance_rank_gold"]
    size = len(rank_error)

    return pd.Series({
        'abs_error': np.mean(abs_error),
        'q-10_rank_error': np.mean(rank_error[:max(1, int(size*0.1))]),
        'q-90_rank_error': np.mean(rank_error[:max(1, int(size*0.9))]),
        'c-1_rank_error': rank_error.iat[0],
        'c-10_rank_error': np.mean(rank_error[:10]),
        'all_rank_error': np.mean(rank_error),
        'walltime': df["walltime"].iat[0]
    })


def _read_csv_tqdm(file, dtype=None, desc=None, leave=True):
    return pd.concat(tqdm(pd.read_csv(file, dtype=dtype, chunksize=1_000_000, usecols=list(dtype.keys())),
                     desc=desc, leave=leave))

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

    if args.stage in ['both', 'preprocess']:
        # Read silver CSV files. We only use seed 0.
        silver_csv_files = {}
        for file in tqdm(sorted(glob.glob(f'{args.persistent_dir}/results/importance_measure/*_s-0_m-i_rs-100.csv.gz')), desc='Parsing and precomputing gold CSVs'):
            filename = path.basename(file)
            dataset, seed, _ = re.match(r'([0-9A-Za-z-]+)_s-(\d+)_m-i_rs-(\d+)', filename).groups()

            # Load corresponding silver file
            tqdm.pandas(desc=f"Precomputing {filename}", leave=False)
            silver_df = _read_csv_tqdm(file, desc=f'Reading {filename}', leave=False, dtype={
                'split': pd.CategoricalDtype(categories=["train", "val", "test"], ordered=True),
                'observation': np.int32,
                'index': np.int32,
                'token': np.int32,
                'importance': np.float32
            })

            silver_csv_files[(dataset[:-4], seed)] = (
                silver_df
                .groupby(['split', 'observation'], observed=True)
                .progress_apply(_precompute_rank)
            )

        # Read CSV files into a dataframe and progressively aggregate the data
        df_partials = []
        df_partials_keys = []
        for file in tqdm(sorted(glob.glob(f'{args.persistent_dir}/results/importance_measure/*_s-0_m-i_rs-*.csv.gz')), desc='Parsing and summarzing CSVs'):
            filename = path.basename(file)
            dataset, seed, riemann_samples = re.match(r'([0-9A-Za-z-]+)_s-(\d+)_m-i_rs-(\d+)', filename).groups()

            df_partial = _read_csv_tqdm(file, desc=f'Reading {filename}', leave=False, dtype={
                'split': pd.CategoricalDtype(categories=["train", "val", "test"], ordered=True),
                'observation': np.int32,
                'index': np.int32,
                'token': np.int32,
                'importance': np.float32,
                'walltime': np.float32
            })
            tqdm.pandas(desc=f"Aggregating {filename}", leave=False)

            df_partials.append(
                df_partial
                .merge(silver_csv_files[(dataset[:-4], seed)], on=["split", "observation", "index"])
                .groupby(['split', 'observation'], observed=True)
                .progress_apply(_aggregate_importance)
            )
            df_partials_keys.append((dataset[:-4], int(seed), int(riemann_samples)))

        df = pd.concat(df_partials, keys=df_partials_keys, names=['dataset', 'seed', 'riemann_samples'])

    if args.stage in ['preprocess']:
        os.makedirs(f'{args.persistent_dir}/pandas', exist_ok=True)
        df.to_pickle(f'{args.persistent_dir}/pandas/riemann.pd.pkl.xz')
        print(df)
    if args.stage in ['plot']:
        df = pd.read_pickle(f'{args.persistent_dir}/pandas/riemann.pd.pkl.xz')

    if args.stage in ['both', 'plot']:
        print(df.head())

        df = (
            df
            .groupby(['dataset', 'seed', 'riemann_samples'])
            .agg({
                "walltime": "sum",
                "abs_error": "mean",
                "q-10_rank_error": "mean",
                "q-90_rank_error": "mean",
                "c-1_rank_error": "mean",
                "c-10_rank_error": "mean",
                "all_rank_error": "mean"
            })
            .groupby(["dataset", "riemann_samples"])
            .agg({
                "walltime": 'mean',
                "abs_error": 'mean',
                "q-10_rank_error": 'mean',
                "q-90_rank_error": 'mean',
                "c-1_rank_error": 'mean',
                "c-10_rank_error":'mean',
                "all_rank_error": 'mean'
            })
            .reset_index()
            .merge(dataset_mapping, on="dataset")
            .drop(columns=['dataset'])
            .rename(columns={
                'dataset_pretty': 'dataset',
                'riemann_samples': 'samples'
            })
            .rename(columns={
                "q-10_rank_error": "q-10",
                "q-90_rank_error": "q-90",
                "c-1_rank_error": "c-1",
                "c-10_rank_error": "c-10",
                "all_rank_error": "all"
            })
        )

        df_latex = (
            df
            .melt(id_vars=["dataset", "samples", "walltime", 'abs_error'],
                value_vars=['q-10', 'q-90', 'c-1', 'c-10', 'all'],
                var_name='ranks',
                value_name='rank-error')
            .pivot(
                index=['dataset', 'samples', 'walltime', 'abs_error'],
                columns=['ranks']
            )
        )

        #print(df_latex)
        os.makedirs(f"{args.persistent_dir}/tables", exist_ok=True)
        df_latex.to_latex(f'{args.persistent_dir}/tables/riemann_samples.tex', escape=False, multirow=True)

        # Generate plot
        df_error = df.melt(id_vars=["dataset", "samples", "walltime", 'abs_error'],
                value_vars=['q-10', 'q-90', 'c-1', 'c-10', 'all'],
                var_name='ranks',
                value_name='rank-error')

        p = (p9.ggplot(df_error.reset_index(), p9.aes(x='samples'))
            + p9.geom_line(p9.aes(y='rank-error', color='ranks'))
            + p9.geom_point(p9.aes(y='rank-error', color='ranks'))
            + p9.facet_grid('dataset ~ .')
            + p9.labs(y='', colour='')
            + p9.scale_y_continuous(labels = lambda ticks: [f'{tick:.0%}' for tick in ticks])
            + p9.scale_x_continuous(name='samples', breaks=range(0, 101, 20))
            + p9.guides(fill=False)
            + p9.theme(plot_margin=0,
                    legend_box = "vertical", legend_position="bottom",
                    text=p9.element_text(size=12))
        )

        p.save(f'{args.persistent_dir}/plots/riemann-error.pdf', width=3.03209, height=7, units='in')
        p.save(f'{args.persistent_dir}/plots/riemann-error.png', width=3.03209, height=7, units='in')

        p = (p9.ggplot(df.reset_index(), p9.aes(x='samples'))
            + p9.geom_line(p9.aes(y='walltime'))
            + p9.geom_point(p9.aes(y='walltime'))
            + p9.facet_grid('dataset ~ .')
            + p9.labs(y='', colour='')
            + p9.scale_y_continuous()
            + p9.scale_x_continuous(name='samples', breaks=range(0, 101, 20))
            + p9.guides(fill=False)
            + p9.theme(plot_margin=0,
                    legend_box = "vertical", legend_position="bottom",
                    text=p9.element_text(size=12))
        )

        p.save(f'{args.persistent_dir}/plots/riemann-walltime.pdf', width=3.03209, height=7, units='in')
        p.save(f'{args.persistent_dir}/plots/riemann-walltime.png', width=3.03209, height=7, units='in')
