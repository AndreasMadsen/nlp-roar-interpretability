
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
from scipy.stats import bootstrap

from nlproar.util import generate_experiment_id

def select_test_metric(partial_df):
    column_name = partial_df.loc[:, 'test_metric'].iat[0]
    return pd.Series({
        'metric': partial_df.loc[:, column_name].iat[0]
    })

def area_between(topks):
    def agg(partial_df):
        partial_df = partial_df.sort_index(level='k').reset_index()
        k = partial_df.loc[:,'k'].to_numpy()
        k_ratio = k / 100
        measure = partial_df.loc[:,'metric'].to_numpy()
        baseline = partial_df.loc[:,'baseline'].to_numpy()

        columns = {}

        for topk in topks:
            mask = (k <= topk)

            y_diff = baseline[mask] - measure[mask]
            x_diff = np.diff(k_ratio[mask])
            areas = (y_diff[1:] + y_diff[0:-1]) * 0.5 * x_diff
            total = np.sum(areas)

            y_diff = baseline[mask] - baseline[-1]
            x_diff = np.diff(k_ratio[mask])
            areas = (y_diff[1:] + y_diff[0:-1]) * 0.5 * x_diff
            max_area = np.sum(areas)

            columns.update({
                f'absolute_area_k-{topk}': total,
                f'relative_area_k-{topk}': np.clip(total / max_area, -1, 1)
            })

        return pd.Series(columns)

    return agg

def format_mean_ci(column_name):
    def formatter(df):
        ret = [
            f'${mean:.1%}^{{+{upper-mean:.1%}}}_{{-{mean-lower:.1%}}}$'.replace('%', '\\%')
            for mean, lower, upper
            in zip(df[f'{column_name}_mean'], df[f'{column_name}_lower'], df[f'{column_name}_upper'])
        ]
        return ret
    return formatter

def ratio_confint_project(column_names):
    """Implementes a ratio-confidence interval

    The idea is to project to logits space, then assume a normal distribution,
    and then project back to the inital space.

    Method proposed here: https://stats.stackexchange.com/questions/263516
    """
    def agg(partial_df):
        summary = dict()
        partial_df = partial_df.reset_index()

        for column_name in column_names:
            x = partial_df.loc[:, column_name].to_numpy()
            logits = scipy.special.logit(x)
            mean = np.mean(logits)
            sem = scipy.stats.sem(logits)

            if np.isnan(sem) or sem == 0:
                lower, upper = mean, mean
            else:
                lower, upper = scipy.stats.t.interval(0.95, len(x) - 1,
                                                    loc=mean,
                                                    scale=sem)

            lower = scipy.special.expit(lower)
            upper = scipy.special.expit(upper)
            mean = scipy.special.expit(mean)

            summary.update({
                f'{column_name}_lower': lower,
                f'{column_name}_mean': mean,
                f'{column_name}_upper': upper,
                f'{column_name}_n': len(x)
            })

        return pd.Series(summary)
    return agg

def ratio_confint(column_names):
    """Implementes a ratio-confidence interval

    This one uses bootstrapping.

    Method proposed here: https://stats.stackexchange.com/questions/263516
    """
    def agg(partial_df):
        summary = dict()
        partial_df = partial_df.reset_index()

        for column_name in column_names:
            x = partial_df.loc[:, column_name].to_numpy()
            mean = np.mean(x)

            if np.all(x[0] == x):
                lower = mean
                upper = mean
            else:
                res = bootstrap((x, ), np.mean, confidence_level=0.95, random_state=np.random.default_rng(0))
                lower = res.confidence_interval.low
                upper = res.confidence_interval.high

            summary.update({
                f'{column_name}_lower': lower,
                f'{column_name}_mean': mean,
                f'{column_name}_upper': upper,
                f'{column_name}_n': len(x)
            })

        return pd.Series(summary)
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
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
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

    model_mapping = pd.DataFrame([
        {'model_type': 'rnn', 'model_type_pretty': 'BiLSTM-Attention'},
        {'model_type': 'roberta', 'model_type_pretty': 'RoBERTa'}
    ])

    recursive_mapping = pd.DataFrame([
        {'recursive': True, 'recursive_pretty': 'Recursive'},
        {'recursive': False, 'recursive_pretty': 'Not Recursive'}
    ])

    importance_measure_mapping = pd.DataFrame([
        {'importance_measure': 'attention', 'importance_measure_pretty': 'Attention'},
        {'importance_measure': 'gradient', 'importance_measure_pretty': 'Gradient'},
        {'importance_measure': 'times-input-gradient', 'importance_measure_pretty': 'Input times Gradient'},
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

        # Duplicate k=0 for 'random' to 'attention', 'gradient', 'integrated-gradient', 'times-input-gradient'
        df_k0 = df.loc[df['k'] == 0]
        df_k0_duplicates = []
        for importance_measure in ['random', 'attention', 'gradient', 'integrated-gradient', 'times-input-gradient']:
            for recursive in [True, False]:
                for strategy in ['count', 'quantile']:
                    df_k0_duplicates.append(
                        df_k0.copy().assign(
                            importance_measure=importance_measure,
                            recursive=recursive,
                            strategy=strategy
                        )
                    )
        df = pd.concat([df.loc[df['k'] != 0], *df_k0_duplicates])

        # Duplicate k=100 for 'random' to 'attention', 'gradient', 'integrated-gradient', 'times-input-gradient'
        df_k100 = df.loc[(df['k'] == 100) & (df['strategy'] == 'quantile')]
        df_k100_duplicates = []
        for importance_measure in ['random', 'attention', 'gradient', 'integrated-gradient', 'times-input-gradient']:
            for recursive in [True, False]:
                df_k100_duplicates.append(
                    df_k100.copy().assign(
                        importance_measure=importance_measure,
                        recursive=recursive,
                        strategy='quantile'
                    )
                )
        df = pd.concat([df.loc[(df['k'] != 100) | (df['strategy'] != 'quantile')], *df_k100_duplicates])

        # Duplicate (random, recursive=False) to recursive=True because
        # this importance measure is model independent
        df = pd.concat([
            df.loc[(df['importance_measure'] != 'random') | (df['recursive'] == False)],
            df.loc[(df['importance_measure'] == 'random') & (df['recursive'] == False)].copy().assign(
                recursive=True
            )
        ])

        # Compute confint and mean for each group
        df = (df.merge(dataset_mapping, on='dataset')
                .drop(['dataset'], axis=1)
                .merge(model_mapping, on='model_type')
                .drop(['model_type'], axis=1)
                .merge(recursive_mapping, on='recursive')
                .drop(['recursive'], axis=1)
                .merge(importance_measure_mapping, on='importance_measure')
                .drop(['importance_measure'], axis=1)
                .groupby([
                    'seed', 'dataset_pretty', 'model_type_pretty',
                    'strategy', 'k', 'recursive_pretty', 'importance_measure_pretty'
                ])
                .apply(select_test_metric)
        )

        print(df)

    if args.stage in ['preprocess']:
        os.makedirs(f'{args.persistent_dir}/pandas', exist_ok=True)
        df.to_pickle(f'{args.persistent_dir}/pandas/roar.pd.pkl.xz')
    if args.stage in ['plot']:
        df = pd.read_pickle(f'{args.persistent_dir}/pandas/roar.pd.pkl.xz')

    if args.stage in ['both', 'plot']:
        df_baseline = (df
            .loc[pd.IndexSlice[:, :, :, :, :, :, ['Random']]]
            .reset_index()
            .drop('importance_measure_pretty', axis=1)
            .rename(columns={'metric': 'baseline'})
        )

        # Generate summary table
        df_latex = (df
            .reset_index()
            .merge(df_baseline, on=['seed', 'dataset_pretty', 'model_type_pretty', 'strategy', 'k', 'recursive_pretty'], how='left')
            .set_index([
                'seed', 'dataset_pretty', 'model_type_pretty',
                'strategy', 'k', 'recursive_pretty', 'importance_measure_pretty'
            ])
            .loc[pd.IndexSlice[:, :, :, 'quantile', :, 'Recursive', ['Attention', 'Gradient', 'Integrated Gradient', 'Input times Gradient']]]
            .reset_index(level=['strategy', 'recursive_pretty'])
            .drop(['strategy', 'recursive_pretty'], axis=1)
            .groupby(['seed', 'dataset_pretty', 'model_type_pretty', 'importance_measure_pretty'])
            .apply(area_between([100]))
            .groupby(['dataset_pretty', 'model_type_pretty', 'importance_measure_pretty'])
            .apply(ratio_confint(['relative_area_k-100']))
            .assign(**{
                'relative_faithfulness_k-100': format_mean_ci('relative_area_k-100')
            })
            .drop(['relative_area_k-100_mean', 'relative_area_k-100_lower', 'relative_area_k-100_upper', 'relative_area_k-100_n'], axis=1)
        )
        print(df_latex)

        os.makedirs(f"{args.persistent_dir}/tables", exist_ok=True)
        (df_latex
            .reset_index()
            .rename(columns={
                'dataset_pretty': 'Dataset',
                'importance_measure_pretty': 'Importance Measure',
                'relative_faithfulness_k-100': 'Faithfulness'
            })
            .pivot(
                index=['Dataset', 'Importance Measure'],
                columns='model_type_pretty',
                values='Faithfulness'
            )
        ).style.to_latex(f'{args.persistent_dir}/tables/faithfulness_metric.tex')

        # Generate result plots
        df_plot = df.reset_index()
        df_plot = (df_plot
            .loc[(df_plot['importance_measure_pretty'] != 'Attention') | (df_plot['model_type_pretty'] == 'BiLSTM-Attention')]
            .groupby(['dataset_pretty', 'model_type_pretty', 'strategy', 'k', 'recursive_pretty', 'importance_measure_pretty'])
            .apply(ratio_confint(['metric']))
        )

        # Generate plot for each strategy
        for strategy in ['count', 'quantile']:
            experiment_id = generate_experiment_id('roar', strategy=strategy, recursive=True)

            p = (p9.ggplot(df_plot.loc[pd.IndexSlice[:, :, strategy, :, 'Recursive', :]].reset_index(), p9.aes(x='k'))
                + p9.geom_ribbon(p9.aes(ymin='metric_lower', ymax='metric_upper', fill='importance_measure_pretty'), alpha=0.35)
                + p9.geom_line(p9.aes(y='metric_mean', color='importance_measure_pretty'))
                + p9.geom_point(p9.aes(y='metric_mean', color='importance_measure_pretty', shape='importance_measure_pretty'))
                + p9.facet_grid('dataset_pretty ~ model_type_pretty', scales='free_y')
                + p9.labs(y='', color='', shape='')
                + p9.scale_y_continuous(labels = lambda ticks: [f'{tick:.0%}' for tick in ticks])
                + p9.scale_color_manual(
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
                p += p9.scale_x_continuous(name='nb. tokens masked', breaks=range(0, 11, 2))
            elif strategy == 'quantile':
                p += p9.scale_x_continuous(name='% tokens masked', breaks=range(0, 101, 20))

            # Save plot, the width is the \linewidth of a collumn in the LaTeX document
            os.makedirs(f'{args.persistent_dir}/plots', exist_ok=True)
            p.save(f'{args.persistent_dir}/plots/{experiment_id}.pdf', width=6.30045 + 0.2, height=7, units='in')
            p.save(f'{args.persistent_dir}/plots/{experiment_id}.png', width=6.30045 + 0.2, height=7, units='in')

        # Generate plot for each model
        for model_type, model_type_pretty in [('rnn', 'BiLSTM-Attention'), ('roberta', 'RoBERTa')]:
            experiment_id = generate_experiment_id(f'roar_{model_type}', strategy='quantile')

            p = (p9.ggplot(df_plot.loc[pd.IndexSlice[:, model_type_pretty, 'quantile', :, :, :]].reset_index(), p9.aes(x='k'))
                + p9.geom_ribbon(p9.aes(ymin='metric_lower', ymax='metric_upper', fill='importance_measure_pretty'), alpha=0.35)
                + p9.geom_line(p9.aes(y='metric_mean', color='importance_measure_pretty'))
                + p9.geom_point(p9.aes(y='metric_mean', color='importance_measure_pretty', shape='importance_measure_pretty'))
                + p9.facet_grid('dataset_pretty ~ recursive_pretty', scales='free_y')
                + p9.labs(y='', color='', shape='')
                + p9.scale_y_continuous(labels = lambda ticks: [f'{tick:.0%}' for tick in ticks])
                + p9.scale_x_continuous(name='% tokens masked', breaks=range(0, 101, 20))
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

            # Save plot, the width is the \linewidth of a collumn in the LaTeX document
            os.makedirs(f'{args.persistent_dir}/plots', exist_ok=True)
            p.save(f'{args.persistent_dir}/plots/{experiment_id}.pdf', width=6.30045 + 0.2, height=7, units='in')
            p.save(f'{args.persistent_dir}/plots/{experiment_id}.png', width=6.30045 + 0.2, height=7, units='in')
