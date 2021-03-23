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
from tqdm import tqdm

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Quantify importance measure sparsity")
parser.add_argument("--persistent-dir",
                    action="store",
                    default=os.path.realpath(os.path.join(thisdir, "..")),
                    type=str,
                    help="Directory where all persistent data will be stored")
parser.add_argument('--num-workers',
                    action='store',
                    default=1,
                    type=int,
                    help='The number of workers to use in data loading')
parser.add_argument('--use-gpu',
                    action='store',
                    default=False,
                    type=bool,
                    help=f'Should GPUs be used')

def _precompute_rank(df):
    return pd.DataFrame({
        'split': df['split'],
        'observation': df['observation'],
        'index': df['index'],
        'token_gold': df['token'],
        'importance_gold': df['importance'],
        'importance_rank_gold':  np.argsort(df['importance'])
    })

def _compute_stats_and_format(col, percentage=False):
    mean = np.mean(col)
    ci = np.abs(scipy.stats.t.ppf(0.025, df=col.size, scale=scipy.stats.sem(col)))

    if percentage:
        return f"${mean:.2%} \\pm {ci:.2%}$".replace(r'%', r'\%')
    else:
        return f"${mean:.2f} \\pm {ci:.2f}$"

def _aggregate_importance(df):
    abs_error = np.abs(df["importance"] - df["importance_rank_gold"])
    rank_error = (np.argsort(df["importance"]) != df["importance_rank_gold"])[::-1]

    return pd.Series({
        'abs_error': np.mean(abs_error),
        'q-10_rank_error': np.mean(rank_error[:int(len(rank_error)*0.1)+1]),
        'c-10_rank_error': np.mean(rank_error[:10]),
        'all_rank_error': np.mean(rank_error),
        'walltime': df["walltime"].iat[0]
    })


def _read_csv_tqdm(file, dtype=None, desc=None, leave=True):
    return pd.concat(tqdm(pd.read_csv(file, dtype=dtype, chunksize=1_000_000, usecols=list(dtype.keys())),
                     desc=desc, leave=leave))

if __name__ == "__main__":
    args = parser.parse_args()
    dataset_mapping = pd.DataFrame([
        {"dataset": "sst", "dataset_pretty": "SST"},
        {"dataset": "snli", "dataset_pretty": "SNLI"},
        {"dataset": "imdb", "dataset_pretty": "IMDB"},
        {"dataset": "babi-2", "dataset_pretty": "bAbI-2"},
        {"dataset": "babi-1", "dataset_pretty": "bAbI-1"},
        {"dataset": "babi-3", "dataset_pretty": "bAbI-3"},
        {"dataset": "mimic-a", "dataset_pretty": "Diabetes"},
        {"dataset": "mimic-d", "dataset_pretty": "Anemia"},
    ])

    # Read gold CSV files
    gold_csv_files = {}
    for file in tqdm(sorted(glob.glob(f'{args.persistent_dir}/results/importance_measure/*_rs-100.csv.gz')), desc='Parsing and precomputing gold CSVs'):
        filename = path.basename(file)
        dataset, seed, _ = re.match(r'([0-9A-Za-z-]+)_s-(\d+)_m-i_rs-(\d+)', filename).groups()

        # Load corresponding gold file
        tqdm.pandas(desc=f"Precomputing {filename}", leave=False)
        gold_df = _read_csv_tqdm(file, desc=f'Reading {filename}', leave=False, dtype={
            'split': pd.CategoricalDtype(categories=["train", "val", "test"], ordered=True),
            'observation': np.int32,
            'index': np.int32,
            'token': np.int32,
            'importance': np.float32
        })

        gold_csv_files[(dataset, seed)] = (
            gold_df
            .groupby(['split', 'observation'], observed=True)
            .progress_apply(_precompute_rank)
        )

    # Read CSV files into a dataframe and progressively aggregate the data
    df_partials = []
    df_partials_keys = []
    for file in tqdm(sorted(glob.glob(f'{args.persistent_dir}/results/importance_measure/*.csv.gz')), desc='Parsing and summarzing CSVs'):
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
            .merge(gold_csv_files[(dataset, seed)], on=["split", "observation", "index"])
            .groupby(['split', 'observation'], observed=True)
            .progress_apply(_aggregate_importance)
        )
        df_partials_keys.append((dataset, int(seed), int(riemann_samples)))

    df = pd.concat(df_partials, keys=df_partials_keys, names=['dataset', 'seed', 'riemann_samples'])

    df_latex = (
        df
        .groupby(['dataset', 'seed', 'riemann_samples'])
        .agg({
            "walltime": "sum",
            "abs_error": "mean",
            "q-10_rank_error": "mean",
            "c-10_rank_error": "mean",
            "all_rank_error": "mean"
        })
        .groupby(["dataset", "riemann_samples"])
        .agg({
            "walltime": partial(_compute_stats_and_format, percentage=False),
            "abs_error": partial(_compute_stats_and_format, percentage=False),
            "q-10_rank_error": partial(_compute_stats_and_format, percentage=True),
            "c-10_rank_error": partial(_compute_stats_and_format, percentage=True),
            "all_rank_error": partial(_compute_stats_and_format, percentage=True)
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
            "c-10_rank_error": "c-10",
            "all_rank_error": "all"
        })
        .melt(id_vars=["dataset", "samples", "walltime", 'abs_error'],
              value_vars=['q-10', 'c-10', 'all'],
              var_name='ranks',
              value_name='rank-error')
        .pivot(
            index=['dataset', 'samples', 'walltime', 'abs_error'],
            columns=['ranks']
        )
    )

    pd.set_option("max_rows", None)
    print(df_latex)
    os.makedirs(f"{args.persistent_dir}/tables", exist_ok=True)
    df_latex.to_latex(f'{args.persistent_dir}/tables/riemann_samples.tex', escape=False, multirow=True)

