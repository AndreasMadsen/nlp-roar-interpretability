
import argparse
import os
import os.path as path

import pandas as pd
import plotnine as p9

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
        df_line = pd.DataFrame([
            {'importance_measure': 'Random', 'metric': 0.90, 'k': 0 },
            {'importance_measure': 'Random', 'metric': 0.86, 'k': 10 },
            {'importance_measure': 'Random', 'metric': 0.72, 'k': 20 },
            {'importance_measure': 'Random', 'metric': 0.61, 'k': 30 },
            {'importance_measure': 'Random', 'metric': 0.55, 'k': 40 },
            {'importance_measure': 'Random', 'metric': 0.50, 'k': 50 },
            {'importance_measure': 'Random', 'metric': 0.45, 'k': 60 },
            {'importance_measure': 'Random', 'metric': 0.38, 'k': 70 },
            {'importance_measure': 'Random', 'metric': 0.29, 'k': 80 },
            {'importance_measure': 'Random', 'metric': 0.24, 'k': 90 },
            {'importance_measure': 'Random', 'metric': 0.20, 'k': 100 },

            {'importance_measure': 'Importance measure', 'metric': 0.90, 'k': 0 },
            {'importance_measure': 'Importance measure', 'metric': 0.70, 'k': 10 },
            {'importance_measure': 'Importance measure', 'metric': 0.45, 'k': 20 },
            {'importance_measure': 'Importance measure', 'metric': 0.32, 'k': 30 },
            {'importance_measure': 'Importance measure', 'metric': 0.29, 'k': 40 },
            {'importance_measure': 'Importance measure', 'metric': 0.28, 'k': 50 },
            {'importance_measure': 'Importance measure', 'metric': 0.27, 'k': 60 },
            {'importance_measure': 'Importance measure', 'metric': 0.24, 'k': 70 },
            {'importance_measure': 'Importance measure', 'metric': 0.20, 'k': 80 },
            {'importance_measure': 'Importance measure', 'metric': 0.20, 'k': 90 },
            {'importance_measure': 'Importance measure', 'metric': 0.20, 'k': 100 },
        ])

        df_area = pd.DataFrame([
            {'area': 'Faithfullness', 'metric_upper': 0.90, 'metric_lower': 0.90, 'k': 0 },
            {'area': 'Faithfullness', 'metric_upper': 0.86, 'metric_lower': 0.70, 'k': 10 },
            {'area': 'Faithfullness', 'metric_upper': 0.72, 'metric_lower': 0.45, 'k': 20 },
            {'area': 'Faithfullness', 'metric_upper': 0.61, 'metric_lower': 0.32, 'k': 30 },
            {'area': 'Faithfullness', 'metric_upper': 0.55, 'metric_lower': 0.29, 'k': 40 },
            {'area': 'Faithfullness', 'metric_upper': 0.50, 'metric_lower': 0.28, 'k': 50 },
            {'area': 'Faithfullness', 'metric_upper': 0.45, 'metric_lower': 0.27, 'k': 60 },
            {'area': 'Faithfullness', 'metric_upper': 0.38, 'metric_lower': 0.24, 'k': 70 },
            {'area': 'Faithfullness', 'metric_upper': 0.29, 'metric_lower': 0.20, 'k': 80 },
            {'area': 'Faithfullness', 'metric_upper': 0.24, 'metric_lower': 0.20, 'k': 90 },
            {'area': 'Faithfullness', 'metric_upper': 0.20, 'metric_lower': 0.20, 'k': 100 },

            {'area': 'Normalizer', 'metric_upper': 0.90, 'metric_lower': 0.20, 'k': 0 },
            {'area': 'Normalizer', 'metric_upper': 0.86, 'metric_lower': 0.20, 'k': 10 },
            {'area': 'Normalizer', 'metric_upper': 0.72, 'metric_lower': 0.20, 'k': 20 },
            {'area': 'Normalizer', 'metric_upper': 0.61, 'metric_lower': 0.20, 'k': 30 },
            {'area': 'Normalizer', 'metric_upper': 0.55, 'metric_lower': 0.20, 'k': 40 },
            {'area': 'Normalizer', 'metric_upper': 0.50, 'metric_lower': 0.20, 'k': 50 },
            {'area': 'Normalizer', 'metric_upper': 0.45, 'metric_lower': 0.20, 'k': 60 },
            {'area': 'Normalizer', 'metric_upper': 0.38, 'metric_lower': 0.20, 'k': 70 },
            {'area': 'Normalizer', 'metric_upper': 0.29, 'metric_lower': 0.20, 'k': 80 },
            {'area': 'Normalizer', 'metric_upper': 0.24, 'metric_lower': 0.20, 'k': 90 },
            {'area': 'Normalizer', 'metric_upper': 0.20, 'metric_lower': 0.20, 'k': 100 },
        ])

    if args.stage in ['both', 'plot']:
        p = (p9.ggplot(mapping=p9.aes(x='k'))
            + p9.geom_ribbon(p9.aes(ymin='metric_lower', ymax='metric_upper', fill='area'), data=df_area, alpha=0.3)
            + p9.geom_line(p9.aes(y='metric', color='importance_measure', shape='importance_measure'), data=df_line)
            + p9.geom_point(p9.aes(y='metric', color='importance_measure', shape='importance_measure'), data=df_line)
            + p9.labs(y='', color='Explanation', shape='Explanation', fill='Area')
            + p9.scale_y_continuous(limits=[0, 1], labels = lambda ticks: [f'{tick:.0%}' for tick in ticks])
            + p9.scale_x_continuous(name='% tokens masked', breaks=range(0, 101, 20))
            + p9.scale_color_manual(
                values = ['#F8766D', '#E76BF3'],
                breaks = ['Importance measure', 'Random']
            )
            + p9.scale_shape_manual(
                values = ['D', 'v'],
                breaks = ['Importance measure', 'Random']
            )
            + p9.scale_fill_manual(
                values = ['#7CAE00', '#00BFC4'],
                breaks = ['Faithfullness', 'Normalizer']
            )
            + p9.theme(plot_margin=0,
                    legend_box = "vertical", legend_position="bottom",
                    text=p9.element_text(size=12))
        )

        # Save plot, the width is the \linewidth of a collumn in the LaTeX document
        os.makedirs(f'{args.persistent_dir}/plots', exist_ok=True)
        p.save(f'{args.persistent_dir}/plots/drawing.pdf', width=3.03209, height=2.08333, units='in')
        p.save(f'{args.persistent_dir}/plots/drawing.png', width=3.03209, height=2.08333, units='in')
