
import pickle
import os
import os.path as path

from tqdm import tqdm
import numpy as np
import torch

from ..util import generate_experiment_id
from ._dataset import Dataset

class ROARDataset(Dataset):
    """Loads a dataset with masking for ROAR."""

    def __init__(self, cachedir, model, base_dataset,
                 k=1, recursive=False, importance_measure='attention',
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
                are: "random", "attention", and "gradient2.
        """
        super().__init__(cachedir, base_dataset.name,
                         base_dataset.tokenizer, batch_size=base_dataset.batch_size,
                         seed=seed, **kwargs)

        self._model = model
        self._base_dataset = base_dataset
        self._k = k
        self._recursive = recursive
        self._importance_measure = importance_measure
        self._read_from_cache = _read_from_cache

        self._basename = generate_experiment_id(base_dataset.name, seed, k, importance_measure, recursive)

        if importance_measure == 'random':
            self._importance_measure_fn = self._importance_measure_random
        elif importance_measure == 'attention':
            self._importance_measure_fn = self._importance_measure_attention
        elif importance_measure == 'gradient':
            self._importance_measure_fn = self._importance_measure_gradient
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
        return torch.tensor(self._np_rng.rand(*observation['sentence'].shape))

    def _importance_measure_attention(self, batch):
        with torch.no_grad():
            _, alpha = self._model(batch)
        return alpha

    def _importance_measure_gradient(self, batch):
        # Make a shallow copy, because batch['sentence'] will be overwritten
        batch = batch.copy()

        # Setup batch to be a one-hot encoded float32 with require_grad. This is neccesary
        # as torch does not allow computing grad w.r.t. to an int-tensor.
        batch['sentence'] = torch.nn.functional.one_hot(batch['sentence'], len(self.vocabulary))
        batch['sentence'] = batch['sentence'].type(torch.float32)
        batch['sentence'].requires_grad = True

        # Compute model
        y, _ = self._model(batch)

        # Select correct label, as we would like gradient of y[correct_label] w.r.t. x
        yc = y[torch.arange(len(batch['label'])), batch['label']]
        # autograd.grad must take a scalar, however we would like $d y_{i,c}/d x_i$
        # to be computed as a batch, meaning for each $i$. To work around this,
        # use that for $g(x) = \sum_i f(x_i)$, we have $d g(x)/d x_{x_i} = d f(x_i)/d x_{x_i}$.
        # The gradient of the sum, is therefore equivalent to the batch_gradient.
        yc_wrt_x, = torch.autograd.grad(torch.sum(yc, axis=0), (batch['sentence'], ))

        # Normalize the vector-gradient per token into one scalar
        return torch.norm(yc_wrt_x, 2, dim=2)

    def _importance_measure_integrated_gradient(self, observation):
        # Implement as x .* (1/k) .* sum([f'((i/k) .* x) for i in range(1, k+1))
        pass

    def _mask_batch(self, batch):
        batch_importance = self._importance_measure_fn(batch)

        masked_batch = []
        with torch.no_grad():
            for importance, observation in zip(batch_importance, self.uncollate(batch)):
                # Trim importance to the observation length
                importance = importance[0:len(observation['sentence'])]

                # Prevent masked tokens from being "removed"
                importance[torch.logical_not(observation['mask'])] = -np.inf

                # Ensure that already "removed" tokens continues to be "removed"
                importance[observation['sentence'] == self.tokenizer.mask_token_id] = np.inf

                # Tokens to remove.
                # Ensure that k does not exceed the number of un-masked tokens, if it does
                # masked tokens will be "removed" too.
                k = torch.minimum(torch.tensor(self._k), torch.sum(observation['mask']))
                _, remove_indices = torch.topk(importance, k=k, sorted=False)

                # "Remove" top-k important tokens
                observation['sentence'][remove_indices] = self.tokenizer.mask_token_id
                masked_batch.append(observation)

        return masked_batch

    def _mask_dataset(self, dataloader, name):
        outputs = []
        for batch in tqdm(dataloader(batch_size=self.batch_size, num_workers=0, shuffle=False),
                          desc=f'Building {name} dataset', leave=False):
            outputs += self._mask_batch(batch)
        return outputs

    def prepare_data(self):
        # Encode data
        if not self._read_from_cache:
            os.makedirs(self._cachedir + '/encoded-roar', exist_ok=True)

            # If we are in a recursive situation, load the k-1 ROAR dataset
            if self._k > 1 and self._recursive:
                base_dataset = ROARDataset(
                    cachedir=self._cachedir,
                    model=self._model,
                    base_dataset=self._base_dataset,
                    k=self._k - 1,
                    recursive=self._recursive,
                    importance_measure=self._importance_measure,
                    seed=self._seed,
                    num_workers=self._num_workers,
                    _read_from_cache=True
                )
            else:
                base_dataset = self._base_dataset

            base_dataset.setup('fit')
            base_dataset.setup('test')

            data = {
                'train': self._mask_dataset(base_dataset.train_dataloader, 'train'),
                'val': self._mask_dataset(base_dataset.val_dataloader, 'val'),
                'test': self._mask_dataset(base_dataset.test_dataloader, 'test')
            }

            with open(f'{self._cachedir}/encoded-roar/{self._basename}.pkl', 'wb') as fp:
                pickle.dump(data, fp)

    def setup(self, stage=None):
        with open(f'{self._cachedir}/encoded-roar/{self._basename}.pkl', 'rb') as fp:
            data = pickle.load(fp)
        if stage == 'fit':
            self._train = data['train']
            self._val = data['val']
        elif stage == 'test':
            self._test = data['test']
        else:
            raise ValueError(f'unexpected setup stage: {stage}')
