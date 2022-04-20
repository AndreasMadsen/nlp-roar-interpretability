
import os
import os.path as path
import pickle

import numpy as np
import torch
import torch.nn as nn

from ..util import generate_experiment_id
from .importance_measures import AttentionImportanceMeasure, \
    GradientImportanceMeasure, IntegratedGradientImportanceMeasure, \
    MutualInformationImportanceMeasure, RandomImportanceMeasure, \
    InputTimesGradientImportanceMeasure
from ._evaluator import ImportanceMeasureEvaluatorUseCache, \
    ImportanceMeasureEvaluatorNoCache, ImportanceMeasureEvaluatorBuildCache

setup_name = {
    'train': 'fit',
    'val': 'fit',
    'test': 'test'
}

class ImportanceMeasure:
    def __init__(self, model, dataset, importance_measure,
                 riemann_samples=50, use_gpu=False, num_workers=4, batch_size=None, seed=0,
                 caching=None, cachedir=None):
        """Create an ImportanceMeasure instance, which explans each obseration in the dataset.

        Args:
            model (MultipleSequenceToClass or SingleSequenceToClass): The model instance to explain
            dataset (Dataset): The dataset comtaining the observations that will be explained.
            importance_measure (str): The importance measure which provides explanations.
            riemann_samples (int, optional): Number of samples used in integrated gradient. Defaults to 50.
            use_gpu (bool, optional): Should a GPU be used for computing explanations.
            num_workers (int, optional): Number of pytourch workers. Defaults to 4.
            batch_size ([type], optional): The batch size.
            seed (int, optional): Random seed, use for random explanation and cache lookup. Defaults to 0.
            caching (None, 'use', 'build'): Should the importance measure use cacheing.
            cachedir ([type], optional): Where should the cache be stored.
        """

        if caching not in [None, 'use', 'build']:
            raise ValueError('caching argument must be either None, "use" or "build"')

        self._dataset = dataset
        self._np_rng = np.random.RandomState(seed)

        self._num_workers = num_workers
        self._batch_size = batch_size
        self._caching = caching
        self._cachedir = cachedir
        self._cachename = generate_experiment_id(f'{dataset.name}_{dataset.model_type}', seed,
                                                 importance_measure=importance_measure,
                                                 riemann_samples=riemann_samples)

        if importance_measure == 'random':
            self._use_gpu = False
            self._importance_measure_fn = RandomImportanceMeasure(
                model, dataset, use_gpu=self._use_gpu, rng=self._np_rng)
        elif importance_measure == 'mutual-information':
            self._use_gpu = use_gpu
            measure = MutualInformationImportanceMeasure(model, dataset, use_gpu=self._use_gpu, rng=self._np_rng)
            measure.precompute(batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False)
            self._importance_measure_fn = measure
        elif importance_measure == 'attention':
            self._use_gpu = use_gpu
            self._importance_measure_fn = AttentionImportanceMeasure(
                model, dataset, use_gpu=self._use_gpu, rng=self._np_rng)
        elif importance_measure == 'gradient':
            self._use_gpu = use_gpu
            self._importance_measure_fn = GradientImportanceMeasure(
                model, dataset, use_gpu=self._use_gpu, rng=self._np_rng)
        elif importance_measure == 'times-input-gradient':
            self._use_gpu = use_gpu
            self._importance_measure_fn = InputTimesGradientImportanceMeasure(
                model, dataset, use_gpu=self._use_gpu, rng=self._np_rng)
        elif importance_measure == 'integrated-gradient':
            self._use_gpu = use_gpu
            self._importance_measure_fn = IntegratedGradientImportanceMeasure(
                model, dataset, riemann_samples=riemann_samples,
                use_gpu=self._use_gpu, rng=self._np_rng)
        else:
            raise ValueError(f'{importance_measure} is not supported')

    def evaluate(self, split):
        """Creates an iterable that provide explanations for each observation in the dataset split.

        Args:
            split: ('train', 'val', 'test') the dataset split to iterate over

        Returns:
            Interable of tubles, (observation, explanation)
        """
        if split not in setup_name:
            raise ValueError(f'split "{split}" is not supported')

        if self._caching is None:
            ImportanceMeasureIterable = ImportanceMeasureEvaluatorNoCache
        elif self._caching == 'build':
            ImportanceMeasureIterable = ImportanceMeasureEvaluatorBuildCache
        elif self._caching == 'use':
            ImportanceMeasureIterable = ImportanceMeasureEvaluatorUseCache

        cache_path = None
        if self._caching is not None:
            cache_path = f'{self._cachedir}/importance-measure/{self._cachename}.{split}.pkl'
            os.makedirs(f'{self._cachedir}/importance-measure', exist_ok=True)

        return ImportanceMeasureIterable(
            self._importance_measure_fn, self._dataset, self._use_gpu, cache_path, split,
            batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False)
