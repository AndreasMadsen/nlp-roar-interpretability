
import glob
import json
import argparse
import os
import os.path as path

from tqdm import tqdm
import pandas as pd
import numpy as np
import scipy
import plotnine as p9

from comp550.util import generate_experiment_id

def select_test_metric(partial_df):
    column_name = partial_df.loc[:, 'test_metric'].iat[0]
    return pd.Series({
        'metric': partial_df.loc[:, column_name].iat[0]
    })

def area_between(partial_df):
    partial_df = partial_df.sort_index(level='k').reset_index()
    k = partial_df.loc[:,'k'].to_numpy() / 100
    measure = partial_df.loc[:,'metric'].to_numpy()
    baseline = partial_df.loc[:,'random'].to_numpy()

    y_diff = baseline - measure
    x_diff = np.diff(k)
    areas = (y_diff[1:] + y_diff[0:-1]) * 0.5 * x_diff
    total = np.sum(areas)

    return pd.Series({
        'area': total
    })

def ratio_confint(column_name):
    """Implementes a ratio-confidence interval

    The idea is to project to logits space, then assume a normal distribution,
    and then project back to the inital space.

    Method proposed here: https://stats.stackexchange.com/questions/263516
    """
    def agg(partial_df):
        x = partial_df.loc[:, column_name]
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

    return agg

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
        {'dataset': 'sst', 'dataset_pretty': 'SST', 'test_metric': 'f1_test'},
        {'dataset': 'snli', 'dataset_pretty': 'SNLI', 'test_metric': 'f1_test'},
        {'dataset': 'imdb', 'dataset_pretty': 'IMDB', 'test_metric': 'f1_test'},
        {'dataset': 'mimic-a', 'dataset_pretty': 'Anemia', 'test_metric': 'f1_test'},
        {'dataset': 'mimic-d', 'dataset_pretty': 'Diabetes', 'test_metric': 'f1_test'},
        {'dataset': 'babi-1', 'dataset_pretty': 'bAbI-1', 'test_metric': 'acc_test'},
        {'dataset': 'babi-2', 'dataset_pretty': 'bAbI-2', 'test_metric': 'acc_test'},
        {'dataset': 'babi-3', 'dataset_pretty': 'bAbI-3', 'test_metric': 'acc_test'},
    ])

    recursive_mapping = pd.DataFrame([
        {'recursive': True, 'recursive_pretty': 'Recursive'},
        {'recursive': False, 'recursive_pretty': 'Not Recursive'}
    ])

    importance_measure_mapping = pd.DataFrame([
        {'importance_measure': 'attention', 'importance_measure_pretty': 'Attention'},
        {'importance_measure': 'gradient', 'importance_measure_pretty': 'Gradient'},
        {'importance_measure': 'integrated-gradient', 'importance_measure_pretty': 'Integrated Gradient'},
        {'importance_measure': 'random', 'importance_measure_pretty': 'Random'}
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
        df = pd.DataFrame(results)

        # Dublicate k=0 for 'random' to 'attention', 'gradient', 'integrated-gradient'
        df_k0 = df.loc[df['k'] == 0]
        df_k0_dublicates = []
        for importance_measure in ['random', 'attention', 'gradient', 'integrated-gradient']:
            for recursive in [True, False]:
                for strategy in ['count', 'quantile']:
                    df_k0_dublicates.append(
                        df_k0.copy().assign(
                            importance_measure=importance_measure,
                            recursive=recursive,
                            strategy=strategy
                        )
                    )
        df = pd.concat([df.loc[df['k'] != 0], *df_k0_dublicates])

        # Dublicate k=100 for 'random' to 'attention', 'gradient', 'integrated-gradient'
        df_k100 = df.loc[(df['k'] == 100) & (df['strategy'] == 'quantile')]
        df_k100_dublicates = []
        for importance_measure in ['random', 'attention', 'gradient', 'integrated-gradient']:
            for recursive in [True, False]:
                df_k100_dublicates.append(
                    df_k100.copy().assign(
                        importance_measure=importance_measure,
                        recursive=recursive,
                        strategy='quantile'
                    )
                )
        df = pd.concat([df.loc[(df['k'] != 100) | (df['strategy'] != 'quantile')], *df_k100_dublicates])

        # Dublicate (random, recursive=False) to recursive=True because
        # this importance measure is model independent
        df = pd.concat([
            df.loc[(df['importance_measure'] != 'random') | (df['recursive'] == False)],
            df.loc[(df['importance_measure'] == 'random') & (df['recursive'] == False)].copy().assign(
                recursive=True
            )
        ])

        # Compute confint and mean for each group
        df = df.merge(dataset_mapping, on='dataset').drop(['dataset'], axis=1)
        df = df.merge(recursive_mapping, on='recursive').drop(['recursive'], axis=1)
        df = df.merge(importance_measure_mapping, on='importance_measure').drop(['importance_measure'], axis=1)
        df = df.groupby(['seed', 'dataset_pretty', 'strategy', 'k', 'recursive_pretty', 'importance_measure_pretty']).apply(select_test_metric)

    if args.stage in ['preprocess']:
        os.makedirs(f'{args.persistent_dir}/pandas', exist_ok=True)
        df.to_pickle(f'{args.persistent_dir}/pandas/roar.pd.pkl.xz')
    if args.stage in ['plot']:
        df = pd.read_pickle(f'{args.persistent_dir}/pandas/roar.pd.pkl.xz')

    if args.stage in ['both', 'plot']:
        # Generate summary table
        df_latex = (df
            .merge(
                (df
                    .loc[pd.IndexSlice[:, :, :, :, :, 'Random']]
                    .reset_index(level='importance_measure_pretty')
                    .drop('importance_measure_pretty', axis=1)
                    .rename(columns={'metric': 'random'})
                ),
                left_index=True, right_index=True)
            .loc[pd.IndexSlice[:, :, 'quantile', :, :, ['Attention', 'Gradient', 'Integrated Gradient']]]
            .groupby(['seed', 'dataset_pretty', 'recursive_pretty', 'importance_measure_pretty']).apply(area_between)
            .apply(lambda col: (col + 1) / 2)
            .groupby(['dataset_pretty', 'recursive_pretty', 'importance_measure_pretty']).apply(ratio_confint('area'))
            .apply(lambda col: (col * 2) - 1)
            .loc[pd.IndexSlice[:, 'Recursive', :]]
            .assign(faithfulness=lambda x: [
                f'${mean:.1%}^{{+{upper-mean:.1%}}}_{{-{mean-lower:.1%}}}$'.replace('%', '\\%')
                for mean, lower, upper
                in zip(x['mean'], x['lower'], x['upper'])
            ])
            .drop(['mean', 'lower', 'upper', 'n'], axis=1)
        )

        os.makedirs(f"{args.persistent_dir}/tables", exist_ok=True)
        df_latex.to_latex(f'{args.persistent_dir}/tables/roar_summary.tex', escape=False, multirow=True)

        # Generate result plots
        df_plot = df.groupby(['dataset_pretty', 'strategy', 'k', 'recursive_pretty', 'importance_measure_pretty']).apply(ratio_confint('metric'))
        for strategy in ['count', 'quantile']:
            experiment_id = generate_experiment_id('roar', strategy=strategy)

            # Generate plot
            p = (p9.ggplot(df_plot.loc[pd.IndexSlice[:, strategy, :, :, :]].reset_index(), p9.aes(x='k'))
                + p9.geom_ribbon(p9.aes(ymin='lower', ymax='upper', fill='importance_measure_pretty'), alpha=0.35)
                + p9.geom_line(p9.aes(y='mean', color='importance_measure_pretty'))
                + p9.geom_point(p9.aes(y='mean', color='importance_measure_pretty'))
                + p9.facet_grid('dataset_pretty ~ recursive_pretty', scales='free_y')
                + p9.labs(y='', colour='')
                + p9.scale_y_continuous(labels = lambda ticks: [f'{tick:.0%}' for tick in ticks])
                + p9.guides(fill=False)
                + p9.theme(plot_margin=0,
                        legend_box = "vertical", legend_position="bottom",
                        text=p9.element_text(size=12)))

            if strategy == 'count':
                p += p9.scale_x_continuous(name='nb. tokens removed', breaks=range(0, 11, 2))
            elif strategy == 'quantile':
                p += p9.scale_x_continuous(name='% tokens removed', breaks=range(0, 101, 20))

            # Save plot, the width is the \linewidth of a collumn in the LaTeX document
            os.makedirs(f'{args.persistent_dir}/plots', exist_ok=True)
            p.save(f'{args.persistent_dir}/plots/{experiment_id}.pdf', width=6.30045 + 0.2, height=7, units='in')
            p.save(f'{args.persistent_dir}/plots/{experiment_id}.png', width=6.30045 + 0.2, height=7, units='in')
