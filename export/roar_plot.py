
import glob
import json
import argparse
import os
import os.path as path

import pandas as pd
import numpy as np
import scipy
import plotnine as p9

import os.path as path

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
        'n': len(x)
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
        {'dataset': 'sst', 'print_name': 'SST', 'test_metric': 'f1_test'},
        {'dataset': 'snli', 'print_name': 'SNLI', 'test_metric': 'f1_test'},
        {'dataset': 'imdb', 'print_name': 'IMDB', 'test_metric': 'f1_test'},
        {'dataset': 'mimic-anemia', 'print_name': 'Anemia', 'test_metric': 'f1_test'},
        {'dataset': 'mimic-diabetes', 'print_name': 'Diabetes', 'test_metric': 'f1_test'},
        {'dataset': 'babi_t-1', 'print_name': 'bAbI-1', 'test_metric': 'acc_test'},
        {'dataset': 'babi_t-2', 'print_name': 'bAbI-2', 'test_metric': 'acc_test'},
        {'dataset': 'babi_t-3', 'print_name': 'bAbI-3', 'test_metric': 'acc_test'},
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

    # Dublicate k=0 for random and top-k masking strategies
    df = pd.concat([
        df.fillna(value={'k': 0, 'masking': 'random'}),
        df[df['masking'].isna()].fillna(value={'k': 0, 'masking': 'top-k'})])
    df = df.loc[df['k'] <= 10, :]
    df = df.merge(dataset_mapping, on='dataset').drop(['dataset'], axis=1)
    # Compute confint and mean for each group
    df = df.groupby(['print_name', 'masking', 'k']).apply(ratio_confint)

    # Generate result table
    pd.set_option('display.max_rows', None)
    print(df)

    # Generate plot
    p = (p9.ggplot(df.reset_index(), p9.aes(x='k'))
        + p9.geom_ribbon(p9.aes(ymin='lower', ymax='upper', fill='masking'), alpha=0.35)
        + p9.geom_line(p9.aes(y='mean', color='masking'))
        + p9.geom_point(p9.aes(y='mean', color='masking'))
        + p9.facet_grid('print_name ~ .', scales='free_y')
        + p9.labs(x='tokens removed', y='', colour='')
        + p9.scale_y_continuous(labels = lambda ticks: [f'{tick:.0%}' for tick in ticks])
        + p9.scale_x_continuous(breaks=range(0, 11, 2))
        + p9.guides(fill=False)
        + p9.theme(plot_margin=0,
                   legend_box = "vertical", legend_position="bottom",
                   text=p9.element_text(size=12)))
    # Save plot, the width is the \linewidth of a collumn in the LaTeX document
    os.makedirs(f'{args.persistent_dir}/plots', exist_ok=True)
    p.save(f'{args.persistent_dir}/plots/roar.pdf', width=3.03209 + 0.2, height=7, units='in')
    p.save(f'{args.persistent_dir}/plots/roar.png', width=3.03209 + 0.2, height=7, units='in')
