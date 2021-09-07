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
from matplotlib import colors

from nlproar.util import generate_experiment_id
from nlproar.dataset import SSTDataset

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--persistent-dir',
                    action='store',
                    default=path.realpath(path.join(thisdir, '..')),
                    type=str,
                    help='Directory where all persistent data will be stored')
parser.add_argument("--seed", action="store", default=0, type=int, help="Random seed")
parser.add_argument("--num-workers",
                    action="store",
                    default=4,
                    type=int,
                    help="The number of workers to use in data loading")
parser.add_argument('--stage',
                    action='store',
                    default='both',
                    type=str,
                    choices=['preprocess', 'plot', 'both'],
                    help='Which export stage should be performed. Mostly just useful for debugging.')

def _read_csv_tqdm(file, dtype=None, desc=None, leave=True):
    return pd.concat(tqdm(pd.read_csv(file, dtype=dtype, chunksize=1_000_000, usecols=list(dtype.keys())),
                     desc=desc, leave=leave))

if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    args, unknown = parser.parse_known_args()

    if args.stage in ['both', 'preprocess']:
        df = _read_csv_tqdm(f'{args.persistent_dir}/results/importance_measure/sst-pre_s-0_m-a_rs-0.csv.gz',
            desc=f'Reading',
            leave=False,
            dtype={
                'split': pd.CategoricalDtype(categories=["train", "val", "test"], ordered=True),
                'observation': np.int32,
                'index': np.int32,
                'token': np.int32,
                'importance': np.float32,
                'walltime': np.float32
            })

    if args.stage in ['preprocess']:
        os.makedirs(f'{args.persistent_dir}/pandas', exist_ok=True)
        df.to_pickle(f'{args.persistent_dir}/pandas/example.pd.pkl.xz')
    if args.stage in ['plot']:
        df = pd.read_pickle(f'{args.persistent_dir}/pandas/example.pd.pkl.xz')

    if args.stage in ['both', 'plot']:
        base_dataset = SSTDataset(
            cachedir=f'{args.persistent_dir}/cache', seed=args.seed, num_workers=args.num_workers
        )
        base_dataset.prepare_data()

        colormap = colors.LinearSegmentedColormap.from_list(
            'importance',
            ['#ffffff', '#fff7ec', '#fee8c8', '#fdd49e', '#fdbb84', '#fc8d59']
        )

        df = (
            df
                .set_index(['split', 'observation', 'index'])
                .loc[pd.IndexSlice['test', 13, :]]
                .sort_index()
                .assign(
                    word=lambda x: x.token.map(dict(enumerate(base_dataset.vocabulary))),
                    importance=lambda x: x.importance / x.importance.max()
                )
        )

        observation = [(row['word'], row['importance']) for _, row in df.iterrows()]
        rank = np.argsort([importance for word, importance in observation])[::-1]

        # Build tikz graphics
        latex = '\\begin{tabular}{p{0.2cm}p{7.8cm}}\n'
        for percentage_i, percentage in enumerate([0, 20, 40]):
            latex += f'{percentage}\\%&'
            indices_masked = rank[0:int(len(observation)*percentage/100)]

            for word_i, (word, importance) in enumerate(observation):
                r,g,b,a = colormap(importance)

                if word_i in indices_masked:
                    latex += f'\colorbox{{rgb,255:red,255; green,255; blue,255}}{{\strut [MASK]}}\\allowbreak'
                elif percentage > 0:
                    latex += f'\colorbox{{rgb,255:red,255; green,255; blue,255}}{{\strut {word}}}\\allowbreak'
                else:
                    latex += f'\colorbox{{rgb,255:red,{r*255:.0f}; green,{g*255:.0f}; blue,{b*255:.0f}}}{{\strut {word}}}\\allowbreak'

            latex += f'\\\\\n'

        latex += '\\end{tabular}\n'

        with open(f'{args.persistent_dir}/plots/example.tex', 'w') as fp:
            fp.write(latex)

