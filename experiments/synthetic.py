
import argparse
import os
import os.path as path
from functools import partial

import numpy as np
import sklearn.linear_model
import pandas as pd
import plotnine as p9

class Dataset:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed=0)
        self.a = np.hstack([
            self.rng.standard_normal((1, 4)),
            np.zeros((1, 12))
        ])
        self.d = self.rng.standard_normal((1, 16))

    def generate(self, samples):
        z = self.rng.standard_normal((samples, 1))
        eta = self.rng.standard_normal((samples, 1))
        eps = self.rng.standard_normal((samples, 1))

        x = (self.a * z) / 10 + self.d * eta + eps / 10
        y = z > 0
        return (x, y[:, 0])

    def train_test(self, samples):
        return (self.generate(samples), self.generate(samples))

class ROAR:
    def __init__(self, train, test, measure, Model, name):
        self.train = train
        self.test = test
        self.Model = Model
        self.measure = measure
        self.name = name

    def __len__(self):
        return 1 + self.train[0].shape[1]

    def __iter__(self):
        train_x, train_y = self.train
        train_x = train_x.copy()
        test_x, test_y = self.test
        test_x = test_x.copy()

        model = self.Model().fit(train_x, train_y)
        train_order = np.argsort(-self.measure(model, train_x), axis=1)
        test_order = np.argsort(-self.measure(model, test_x), axis=1)

        yield model.score(test_x, test_y)

        for mask_feature_i in range(train_order.shape[1]):
            train_x[np.arange(train_x.shape[0]), train_order[:, mask_feature_i]] = 0
            test_x[np.arange(test_x.shape[0]), test_order[:, mask_feature_i]] = 0
            yield self.Model().fit(train_x, train_y).score(test_x, test_y)

class RecursiveROAR:
    def __init__(self, train, test, measure, Model, name):
        self.train = train
        self.test = test
        self.Model = Model
        self.measure = measure
        self.name = name

    def __len__(self):
        return 1 + self.train[0].shape[1]

    def __iter__(self):
        train_x, train_y = self.train
        train_x = train_x.copy()
        test_x, test_y = self.test
        test_x = test_x.copy()

        model = self.Model().fit(train_x, train_y)
        yield model.score(*self.test)

        train_masked = None
        test_masked = None
        for i in range(16):
            train_importance = self.measure(model, train_x)
            test_importance = self.measure(model, test_x)

            if i == 0:
                train_masked = np.zeros_like(train_importance, dtype=np.bool_)
                test_masked = np.zeros_like(test_importance, dtype=np.bool_)

            train_importance[train_masked] = float('-inf')
            test_importance[test_masked] = float('-inf')

            train_mask_feature = np.argmax(train_importance, axis=1)
            test_mask_feature = np.argmax(test_importance, axis=1)

            train_masked[np.arange(train_masked.shape[0]), train_mask_feature] = True
            test_masked[np.arange(test_masked.shape[0]), test_mask_feature] = True
            train_x[train_masked] = 0
            test_x[test_masked] = 0

            model = self.Model().fit(train_x, train_y)
            yield model.score(test_x, test_y)

class RandomImportance:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed=0)

    def __call__(self, model, observations):
        return self.rng.uniform(0, 1, observations.shape)

class WeightImportance:
    def __call__(self, model, observations):
        importance = np.abs(model.coef_.ravel())
        return np.tile(importance, (observations.shape[0], 1))

class BestImportance:
    def __call__(self, model, observations):
        importance = np.asarray([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        return np.tile(importance, (observations.shape[0], 1))

class WorstImportance:
    def __call__(self, model, observations):
        importance = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        return np.tile(importance, (observations.shape[0], 1))

class LogisticRegression:
    def __init__(self, seed=0):
        self.model = sklearn.linear_model.LogisticRegression(penalty='none', random_state=seed)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def score(self, X, y):
        return self.model.score(X, y)

    @property
    def coef_(self):
        return self.model.coef_

class LinearRegression:
    def __init__(self, seed=0):
        self.model = sklearn.linear_model.LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y.astype(X.dtype))
        return self

    def score(self, X, y):
        y_pred = self.model.predict(X) > 0.5
        return np.mean(y_pred == y)

    @property
    def coef_(self):
        return self.model.coef_

thisdir = path.dirname(path.realpath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--persistent-dir',
                    action='store',
                    default=path.realpath(path.join(thisdir, '..')),
                    type=str,
                    help='Directory where all persistent data will be stored')
parser.add_argument('--seed',
                    action='store',
                    default=0,
                    type=int,
                    help='Seed to use.')

if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    args, unknown = parser.parse_known_args()

    data = []

    for seed in [args.seed]:
        Model = partial(LogisticRegression, seed=seed)
        dataset = Dataset(seed=seed)
        train = dataset.generate(50000)
        test = dataset.generate(10000)
        for strategy in [
            ROAR(train, test, RandomImportance(seed=seed), Model, name='Random'),
            ROAR(train, test, WeightImportance(), Model, name='ROAR'),
            RecursiveROAR(train, test, WeightImportance(), Model, name='Recursive ROAR'),
            ROAR(train, test, BestImportance(), Model, name='Best case'),
            ROAR(train, test, WorstImportance(), Model, name='Worst case')
        ]:
            for features_removed, test_acc in enumerate(strategy):
                data.append({'seed': seed, 'measure': strategy.name, 'features_removed': features_removed, 'test_acc': test_acc })

    df = pd.DataFrame(data)

    p = (p9.ggplot(df, p9.aes(x='features_removed', y='test_acc', color='measure', linetype='measure'))
        + p9.geom_line()
        + p9.geom_point()
        + p9.labs(y='Test Accuracy')
        + p9.scale_y_continuous(limits=(0.4, 1), labels = lambda ticks: [f'{tick:.0%}' for tick in ticks])
        + p9.scale_x_continuous(name='Number of features masked', breaks=range(0, 17, 2))
        + p9.scale_linetype_manual(
            values = ['solid', 'solid', 'dashed', 'solid', 'solid'],
            breaks = ['Random', 'ROAR', 'Recursive ROAR', 'Best case', 'Worst case']
        )
        + p9.guides(linetype=None, color=p9.guide_legend(nrow=2, byrow=True))
        + p9.theme(plot_margin=0,
                   legend_title=p9.element_blank(),
                   legend_box_margin=0, legend_box_spacing=0.7,
                   legend_box = "vertical", legend_position="bottom",
                   text=p9.element_text(size=12))
    )

    # Save plot, the width is the \linewidth of a collumn in the LaTeX document
    os.makedirs(f'{args.persistent_dir}/plots', exist_ok=True)
    p.save(f'{args.persistent_dir}/plots/synthetic.pdf', width=3.03209, height=3, units='in')
    p.save(f'{args.persistent_dir}/plots/synthetic.png', width=3.03209, height=3, units='in')
