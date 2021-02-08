
import pickle
import os
import os.path as path

from tqdm import tqdm
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class ROARDataset(pl.LightningDataModule):
    """Loads a dataset with masking for ROAR."""

    def __init__(self, cachedir, model, base_dataset,
                 k=1, recursive=False, importance_measure='attention',
                 batch_size=128, seed=0, num_workers=4):
        """
        Args:
            model: The model to use to determine which tokens to mask.
            recursive: Should roar masking be applied recursively. This only affects
                the filename of the saved dataset. The base_dataset is assumed to be
                for k-1.
            base_dataset: The dataset to apply masking to.
            k (int): The number of tokens to mask for each instance in the
                dataset.
            importance_measure (str): Which importance measure to use. Supported values
                are: "random" and "attention".
        """
        super().__init__()

        self._cachedir = cachedir
        self._model = model
        self._base_dataset = base_dataset
        self._k = k
        self._recursive = recursive
        self._batch_size = batch_size
        self._seed = seed
        self._num_workers = num_workers

        self._basename = f'{self.name}_roar_k-{k}_r-{int(recursive)}_m-{importance_measure[0]}'
        self._rng = np.random.RandomState(self._seed)

        if importance_measure == 'random':
            self._importance_measure = self._importance_measure_random
        elif importance_measure == 'attention':
            self._importance_measure = self._importance_measure_attention
        else:
            raise ValueError(f'{importance_measure} is not supported')

    @property
    def tokenizer(self):
        return self._base_dataset.tokenizer

    @property
    def label_names(self):
        return self._base_dataset.label_names

    @property
    def name(self):
        return self._base_dataset.name

    def embedding(self):
        return self._base_dataset.embedding()

    def collate(self, observations):
        return self._base_dataset.collate(observations)

    def uncollate(self, observations):
        return self._base_dataset.uncollate(observations)

    def _importance_measure_random(self, batch):
        return torch.tensor(self._rng.rand(*batch['sentence'].shape))

    def _importance_measure_attention(self, batch):
        with torch.no_grad():
            _, alpha = self._model(batch)
        return alpha

    def _mask_batch(self, batch):
        batch_importance = self._importance_measure(batch)

        with torch.no_grad():
            for observation, importance in zip(self._base_dataset.uncollate(batch), batch_importance):
                importance = importance[:observation['length']]

                # Prevent masked tokens from being "removed"
                importance[torch.logical_not(observation['mask'])] = -np.inf

                # Ensure that already "removed" tokens continues to be "removed"
                importance[observation['sentence'] == self.tokenizer.mask_token_id] = np.inf

                # Tokens to remove.
                # Ensure that k does not exceed the number of un-masked tokens, if it does
                # masked tokens will be "removed" too.
                k = torch.maximum(torch.tensor(self._k), torch.sum(observation['mask']))
                _, remove_indices = torch.topk(importance, k=k, sorted=False)

                # "Remove" top-k important tokens
                observation['sentence'][remove_indices] = self.tokenizer.mask_token_id
                yield observation

    def _mask_data(self, dataloader, name):
        outputs = []
        for batch in tqdm(dataloader, desc=f'Building {name} dataset', leave=False):
            outputs.extend(self._mask_batch(batch))
        return outputs

    def prepare_data(self):
        # Encode data
        if not path.exists(f'{self._cachedir}/encoded/{self._basename}.pkl'):
            os.makedirs(self._cachedir + '/encoded', exist_ok=True)

            self._base_dataset.setup('fit')
            self._base_dataset.setup('test')

            data = {
                'train': self._mask_data(self._base_dataset.train_dataloader(), 'train'),
                'val': self._mask_data(self._base_dataset.val_dataloader(), 'val'),
                'test': self._mask_data(self._base_dataset.val_dataloader(), 'test')
            }

            with open(f'{self._cachedir}/encoded/{self._basename}.pkl', 'wb') as fp:
                pickle.dump(data, fp)

    def setup(self, stage=None):
        with open(f'{self._cachedir}/encoded/{self._basename}.pkl', 'rb') as fp:
            data = pickle.load(fp)
        if stage == 'fit':
            self._train = data['train']
            self._val = data['val']
        elif stage == 'test':
            self._test = data['test']
        else:
            raise ValueError(f'unexpected setup stage: {stage}')

    def train_dataloader(self):
        return DataLoader(
            self._train,
            batch_size=self._batch_size, collate_fn=self.collate,
            num_workers=self._num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(
            self._val,
            batch_size=self._batch_size, collate_fn=self.collate,
            num_workers=self._num_workers)

    def test_dataloader(self):
        return DataLoader(
            self._test,
            batch_size=self._batch_size, collate_fn=self.collate,
            num_workers=self._num_workers)
