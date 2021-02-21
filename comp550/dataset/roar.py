from typing import TypeVar, Generic, Union

import pickle
import os
import os.path as path

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from ..util import generate_experiment_id
from ._dataset import Dataset, SequenceBatch

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

class ImportanceMeasureModule(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

class AttentionImportanceMeasureModule(ImportanceMeasureModule):
    def forward(self, batch: SequenceBatch) -> torch.Tensor:
        with torch.no_grad():
            _, alpha, _ = self.model(batch)
        return alpha

class GradientImportanceMeasureModule(ImportanceMeasureModule):
    def forward(self, batch: SequenceBatch) -> torch.Tensor:
        # Compute model
        y, _, embedding = self.model(batch)
        # Select correct label, as we would like gradient of y[correct_label] w.r.t. x
        yc = y[torch.arange(batch.label.numel()), batch.label]

        # autograd.grad must take a scalar, however we would like $d y_{i,c}/d x_i$
        # to be computed as a batch, meaning for each $i$. To work around this,
        # use that for $g(x) = \sum_i f(x_i)$, we have $d g(x)/d x_{x_i} = d f(x_i)/d x_{x_i}$.
        # The gradient of the sum, is therefore equivalent to the batch_gradient.
        yc_batch = torch.sum(yc, dim=0)

        with torch.no_grad():
            yc_wrt_embedding, = torch.autograd.grad([yc_batch], (embedding, ))
            if yc_wrt_embedding is None:
                raise ValueError('Could not compute gradient')

            # We need the gradient wrt. x. However, to compute that directly with .grad would
            # require the model input to be a one_hot encoding. Creating a one_hot encoding
            # is very memory inefficient. To avoid that, manually compute the gradient wrt. x
            # based on the gradient yc_wrt_embedding.
            # yc_wrt_x = yc_wrt_emb @ emb_wrt_x = yc_wrt_emb @ emb_matix.T
            embedding_matrix_t = torch.transpose(self.model.embedding_matrix, 0, 1)
            yc_wrt_x = torch.matmul(yc_wrt_embedding, embedding_matrix_t)

            # Normalize the vector-gradient per token into one scalar
            return torch.norm(yc_wrt_x, p=2, dim=2)

class ROARDataset(Dataset):
    """Loads a dataset with masking for ROAR."""

    def __init__(self, cachedir, model, base_dataset,
                 k=1, strategy='count',
                 recursive=False, recursive_step_size=1,
                 importance_measure='attention',
                 build_batch_size=None, use_gpu=False,
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
        self._build_batch_size = build_batch_size
        self._use_gpu = use_gpu
        self._read_from_cache = _read_from_cache

        self._basename = generate_experiment_id(base_dataset.name, seed,
                                                k=k,
                                                strategy=strategy,
                                                importance_measure=importance_measure,
                                                recursive=recursive)

        if importance_measure == 'random':
            self._importance_measure_calc = None
            self._importance_measure_fn = self._importance_measure_random
        elif importance_measure == 'attention':
            self._importance_measure_calc = torch.jit.script(AttentionImportanceMeasureModule(self._model))
            self._importance_measure_fn = self._importance_measure_attention
        elif importance_measure == 'gradient':
            self._importance_measure_calc = torch.jit.script(GradientImportanceMeasureModule(self._model))
            self._importance_measure_fn = self._importance_measure_gradient
        elif importance_measure == 'integrated-gradient':
            self._importance_measure_fn = self._importance_measure_integrated_gradient
        else:
            raise ValueError(f'{importance_measure} is not supported')

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

    def _importance_measure_random(self, observation):
        return torch.tensor(self._np_rng.rand(*observation.sentence.shape))

    def _importance_measure_attention(self, batch):
        return self._importance_measure_calc(batch.cuda() if self._use_gpu else batch)

    def _importance_measure_gradient(self, batch):
        return self._importance_measure_calc(batch.cuda() if self._use_gpu else batch)

    def _importance_measure_integrated_gradient(self, observation):

        num_intervals = 20

        observations = [observation]*num_intervals
        batch = self.collate(observations)
        batch['sentence'] = torch.nn.functional.one_hot(batch['sentence'], len(self.vocabulary))
        batch['sentence'] = batch['sentence'].type(torch.float32)
        batch['sentence'].requires_grad = True

        interval = torch.arange(1, num_intervals + 1)/num_intervals
        batch['sentence'] = batch['sentence'] * interval.unsqueeze(1).unsqueeze(2)

        for k in batch.keys():
            if k is not 'length':
                batch[k] = batch[k].cuda()
        
        y, _ = self._model(batch)
        yc = y[:, observation['label']]

        yc_wrt_x = torch.autograd.grad(yc.sum(), (batch['sentence']))[0]
        yc_wrt_x = yc_wrt_x.sum(dim=0)
        yc_wrt_x = (1/num_intervals)*batch['sentence'][-1]*yc_wrt_x

        return torch.norm(yc_wrt_x, 2, dim=1)

    def _mask_batch(self, batch):
        batch_importance = self._importance_measure_fn(batch)
        if self._use_gpu:
            batch_importance = batch_importance.cpu()

        masked_batch = []
        with torch.no_grad():
            for importance, observation in zip(batch_importance, self.uncollate(batch)):
                # Trim importance to the observation length
                importance = importance[0:observation['length']]

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

                masked_batch.append({ key: val.tolist() for key, val in observation.items() })

        return masked_batch

    def _mask_dataset(self, dataloader, name):
        outputs = []
        for batch in tqdm(dataloader(batch_size=self._build_batch_size,
                                     num_workers=min(self._num_workers, 1),
                                     shuffle=False),
                          desc=f'Building {name} dataset', leave=False):
            outputs += self._mask_batch(batch)
        return outputs

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
                    build_batch_size=self._build_batch_size,
                    seed=self._seed,
                    num_workers=self._num_workers,
                    _read_from_cache=True
                )
            else:
                base_dataset = self._base_dataset

            # Mask each dataset split according to the importance measure
            if self._use_gpu:
                self._model.cuda()

            base_dataset.setup('fit')
            train = self._mask_dataset(base_dataset.train_dataloader, 'train')
            val = self._mask_dataset(base_dataset.val_dataloader, 'val')
            base_dataset.clean('fit')

            base_dataset.setup('test')
            test = self._mask_dataset(base_dataset.test_dataloader, 'test')
            base_dataset.clean('test')

            # save data
            with open(f'{self._cachedir}/encoded-roar/{self._basename}.pkl', 'wb') as fp:
                pickle.dump({'train': train, 'val': val, 'test': test}, fp)

            # Free the refcount to the model, as it is not required anymore
            del self._model
            del self._importance_measure_calc

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
