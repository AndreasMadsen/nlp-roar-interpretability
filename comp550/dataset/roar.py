import pickle
import os
import os.path as path

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from ._dataset import Dataset
from ..util import generate_experiment_id
from ..explain import ImportanceMeasure

lookup_dtype = {
    'sentence': torch.int64,
    'length': torch.int64,
    'mask': torch.bool,
    'sentence_aux': torch.int64,
    'sentence_aux_length': torch.int64,
    'sentence_aux_mask': torch.bool,
    'label': torch.int64,
    'index': torch.int64
}

class ROARDataset(Dataset):
    """Loads a dataset with masking for ROAR."""

    def __init__(self, cachedir, model, base_dataset,
                 k=1, strategy='count',
                 recursive=False, recursive_step_size=1,
                 importance_measure='attention',
                 riemann_samples=20,
                 build_batch_size=None, importance_caching=None,
                 use_gpu=False,
                 seed=0, _read_from_cache=False, **kwargs):
        """
        Args:
            model: The model to use to determine which tokens to mask.
            base_dataset: The dataset to apply masking to.
            k (int): The number of tokens to mask for each instance in the
                dataset.
            recursive (bool): Should roar masking be applied recursively.
                If recursive is used, the model should be trained for k-1
                and the base_dataset should be for k=0.
            importance_measure (str): Which importance measure to use. Supported values
                are: "random", "attention", "gradient" and "integrated-gradient".
        """
        super().__init__(cachedir, base_dataset.name,
                         base_dataset.tokenizer, batch_size=base_dataset.batch_size,
                         seed=seed, **kwargs)

        if strategy not in ['count', 'quantile']:
            raise ValueError(f'The "{strategy}" strategy is not supported')

        self._model = model
        self._base_dataset = base_dataset
        self._k = k
        self._strategy = strategy
        self._recursive = recursive
        self._recursive_step_size = recursive_step_size
        self._importance_measure = importance_measure
        self._riemann_samples = riemann_samples
        self._build_batch_size = build_batch_size
        self._importance_caching = importance_caching
        self._use_gpu = use_gpu
        self._read_from_cache = _read_from_cache

        self._basename = generate_experiment_id(base_dataset.name, seed,
                                                k=k,
                                                strategy=strategy,
                                                importance_measure=importance_measure,
                                                recursive=recursive)

        if _read_from_cache:
            if not path.exists(f'{self._cachedir}/encoded-roar/{self._basename}.pkl'):
                raise IOError((f'The ROAR dataset "{self._basename}", does not exists.'
                               f' For optimization reasons it has been decided that the k-1 ROAR dataset must exist'))

    @property
    def label_names(self):
        return self._base_dataset.label_names

    def embedding(self):
        return self._base_dataset.embedding()

    def collate(self, observations):
        return self._base_dataset.collate(observations)

    def uncollate(self, observations):
        return self._base_dataset.uncollate(observations)

    def _mask_dataset(self, importance_measure, split):
        for observation, importance in tqdm(importance_measure.evaluate(split),
                                            desc=f'Building {split} dataset', leave=False):
            with torch.no_grad():
                # Prevent masked tokens from being "removed"
                importance[torch.logical_not(observation['mask'])] = -np.inf

                # Ensure that already "removed" tokens continues to be "removed"
                importance[observation['sentence'] == self.tokenizer.mask_token_id] = np.inf

                # Tokens to remove.
                # Ensure that k does not exceed the number of un-masked tokens, if it does
                # masked tokens will be "removed" too.
                no_attended_elements = torch.sum(observation['mask'])
                if self._strategy == 'count':
                    k = torch.minimum(torch.tensor(self._k), no_attended_elements)
                elif self._strategy == 'quantile':
                    k = (torch.tensor(self._k / 100) * no_attended_elements).int()

                # "Remove" top-k important tokens
                _, remove_indices = torch.topk(importance, k=k, sorted=False)
                observation['sentence'][remove_indices] = self.tokenizer.mask_token_id

                yield { key: val.tolist() for key, val in observation.items() }

    def prepare_data(self):
        # Encode data
        if not self._read_from_cache:
            os.makedirs(self._cachedir + '/encoded-roar', exist_ok=True)

            # If we are in a recursive situation, load the k-1 ROAR dataset
            if self._k > self._recursive_step_size and self._recursive:
                base_dataset = ROARDataset(
                    cachedir=self._cachedir,
                    model=self._model,
                    base_dataset=self._base_dataset,
                    k=self._k - self._recursive_step_size,
                    strategy=self._strategy,
                    recursive=self._recursive,
                    recursive_step_size=self._recursive_step_size,
                    importance_measure=self._importance_measure,
                    riemann_samples=self._riemann_samples,
                    build_batch_size=self._build_batch_size,
                    seed=self._seed,
                    num_workers=self._num_workers,
                    _read_from_cache=True
                )
            else:
                base_dataset = self._base_dataset


            importance_measure = ImportanceMeasure(
                self._model, base_dataset, self._importance_measure,
                riemann_samples=self._riemann_samples,
                use_gpu=self._use_gpu,
                num_workers=min(self._num_workers, 1),
                batch_size=self._build_batch_size,
                seed=self._seed,
                caching=self._importance_caching,
                cachedir=self._cachedir,
                cachename=generate_experiment_id(
                    base_dataset.name, self._seed,
                    k=self._k - self._recursive_step_size if recursive else 0,
                    strategy=self._strategy,
                    importance_measure=self._importance_measure,
                    recursive=self._recursive
                )
            )

            # save data
            with open(f'{self._cachedir}/encoded-roar/{self._basename}.pkl', 'wb') as fp:
                pickle.dump({
                    'train': list(self._mask_dataset(importance_measure, 'train')),
                    'val': list(self._mask_dataset(importance_measure, 'val')),
                    'test': list(self._mask_dataset(importance_measure, 'test'))
                }, fp)

            # Free the refcount to the model, as it is not required anymore
            del self._model

            # Because the self._model ref no longer exists, the dataset can't be rebuild
            self._read_from_cache = True

    def _pickle_data_to_torch_data(self, data):
        return [{
            key: torch.tensor(val, dtype=lookup_dtype[key]) for key, val in x.items()
        } for x in data]

    def setup(self, stage=None):
        with open(f'{self._cachedir}/encoded-roar/{self._basename}.pkl', 'rb') as fp:
            data = pickle.load(fp)
        if stage == 'fit':
            self._train = self._pickle_data_to_torch_data(data['train'])
            self._val = self._pickle_data_to_torch_data(data['val'])
        elif stage == 'test':
            self._test = self._pickle_data_to_torch_data(data['test'])
        else:
            raise ValueError(f'unexpected setup stage: {stage}')
