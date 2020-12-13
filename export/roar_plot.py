
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

def ratio_confint(partial_df, column_name='f1_test'):
    """Implementes a ratio-confidence interval

    The idea is to project to logits space, then assume a normal distribution,
    and then project back to the inital space.

    Method proposed here: https://stats.stackexchange.com/questions/263516
    """
    x = partial_df[column_name]
    logits = scipy.special.logit(x)
    mean = np.mean(logits)
    lower, upper = scipy.stats.t.interval(0.95, len(x) - 1,
                                          loc=mean,
                                          scale=scipy.stats.sem(logits))

    return pd.Series({
        'lower': scipy.special.expit(lower),
        'mean': scipy.special.expit(mean),
        'upper': scipy.special.expit(upper),
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

    # Read JSON files into dataframe
    results = []
    for file in glob.glob(f'{args.persistent_dir}/results/*.json'):
        with open(file, 'r') as fp:
            results.append(json.load(fp))
    df = pd.DataFrame(results)

    # Dublicate k=0 for random and top-k masking strategies
    df = pd.concat([
        df.fillna(value={'k': 0, 'masking': 'random'}),
        df[df['masking'].isna()].fillna(value={'k': 0, 'masking': 'top-k'})])

    # Make the facet pretty
    df.replace({'dataset': {
        'sst': 'SST',
        'snli': 'SNLI',
        'mimic-anemia': 'Anemia',
        'mimic-diabetes': 'Diabetes'
    }}, inplace=True)
    # Compute confint and mean for each group
    df = df.groupby(['dataset', 'masking', 'k']).apply(ratio_confint)

    # Generate result table
    pd.set_option('display.max_rows', None)
    print(df)

    # Generate plot
    p = (p9.ggplot(df.reset_index(), p9.aes(x='k'))
        + p9.geom_ribbon(p9.aes(ymin='lower', ymax='upper', fill='masking'), alpha=0.35)
        + p9.geom_line(p9.aes(y='mean', color='masking'))
        + p9.geom_point(p9.aes(y='mean', color='masking'))
        + p9.facet_grid('dataset ~ .', scales='free_y')
        + p9.labs(x='tokens removed', y='F1-Score', colour='')
        + p9.scale_y_continuous(labels = lambda ticks: [f'{tick:.1%}' for tick in ticks])
        + p9.guides(fill=False)
        + p9.theme(plot_margin=0,
                   legend_box = "vertical", legend_position="bottom",
                   text=p9.element_text(size=12)))
    # Save plot, the width is the \linewidth of a collumn in the LaTeX document
    os.makedirs(f'{args.persistent_dir}/plots', exist_ok=True)
    p.save(f'{args.persistent_dir}/plots/roar.pdf', width=3.03209, height=4, units='in')
    p.save(f'{args.persistent_dir}/plots/roar.png', width=3.03209, height=4, units='in')
