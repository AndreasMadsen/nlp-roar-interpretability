
import torch
import pickle

setup_name = {
    'train': 'fit',
    'val': 'fit',
    'test': 'test'
}

class ImportanceMeasureEvaluator:
    def __init__(self, importance_measure_fn, dataset, use_gpu, cache_path, split, **kwargs):
        self._importance_measure_fn = importance_measure_fn
        self._dataset = dataset
        self._use_gpu = use_gpu
        self._cache_path = cache_path
        self._split = split
        self._kwargs = kwargs

        self._was_setup = self._dataset.is_setup(setup_name[self._split])
        if not self._was_setup:
            self._dataset.setup(setup_name[self._split])

        self._length = self._dataset.num_of_observations(self._split)

    def _finalize(self):
        if not self._was_setup:
            self._dataset.clean(setup_name[self._split])

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        return self._length

class ImportanceMeasureEvaluatorNoCache(ImportanceMeasureEvaluator):
    def __iter__(self):
        for batch in self._dataset.dataloader(self._split, **self._kwargs):
            batch_importance = self._importance_measure_fn(batch.cuda() if self._use_gpu else batch)
            batch_importance = batch_importance.cpu() if self._use_gpu else batch_importance

            yield from (
                (observation, importance[0:observation['length'].tolist()])
                for observation, importance
                in zip(self._dataset.uncollate(batch), batch_importance)
            )

        self._finalize()

class ImportanceMeasureEvaluatorBuildCache(ImportanceMeasureEvaluatorNoCache):
    def __iter__(self):
        cache = dict()

        for observation, importance in super().__iter__():
            cache[observation['index'].tolist()] = importance.tolist()
            yield (observation, importance)

        with open(self._cache_path, 'wb') as fp:
            pickle.dump(cache, fp)

class ImportanceMeasureEvaluatorUseCache(ImportanceMeasureEvaluator):
    def __iter__(self):
        with open(self._cache_path, 'rb') as fp:
            cache = pickle.load(fp)

        for batch in self._dataset.dataloader(self._split, **self._kwargs):
            yield from (
                (observation, torch.tensor(cache[observation['index'].tolist()], dtype=torch.float32))
                for observation
                in self._dataset.uncollate(batch)
            )

        self._finalize()
