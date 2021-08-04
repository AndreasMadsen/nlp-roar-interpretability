
import os
import os.path as path
import pickle

import numpy as np
import torch
import torch.nn as nn

from ..dataset import SequenceBatch
from ..util import generate_experiment_id

setup_name = {
    'train': 'fit',
    'val': 'fit',
    'test': 'test'
}

class ImportanceMeasureModule(nn.Module):
    def __init__(self, model, dataset, use_gpu, rng):
        super().__init__()
        self.model = model.cuda() if use_gpu else model
        self.model.flatten_parameters()
        self.dataset = dataset
        self.device = torch.device('cuda' if use_gpu else 'cpu')
        self.rng = rng

class RandomImportanceMeasure(ImportanceMeasureModule):
    def forward(self, batch: SequenceBatch) -> torch.Tensor:
        return torch.tensor(self.rng.rand(*batch.sentence.shape))

class MutualInformationImportanceMeasure(ImportanceMeasureModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mutual_information = torch.zeros(len(self.dataset.vocabulary), len(self.dataset.label_names))

    def precompute(self, *args, **kwargs):
        # Prepare dataset
        was_setup = self.dataset.is_setup('fit')
        if not was_setup:
            self.dataset.setup('fit')

        # Count number (word, label) pairs. Note that the same word appearing multiple times
        #   in one sentences, is just counted as one word.
        # Start counters, with "1" to indicate there is a fake document with all words for each class.
        #   This is to avoid divide-by-zero issues, which is a limitation of KL-divergence / Mutual Information.
        N_docs = torch.tensor(self.dataset.num_of_observations('train') + len(self.dataset.label_names), dtype=torch.int32)
        N_docs_label_1 = torch.ones(1, len(self.dataset.label_names), dtype=torch.int32)
        N_word_1_label_1 = torch.ones(len(self.dataset.vocabulary), len(self.dataset.label_names), dtype=torch.int32)

        for batch in self.dataset.dataloader('train', *args, **kwargs):
            for observation in self.dataset.uncollate(batch):
                words = torch.bincount(observation['sentence'], minlength=len(self.dataset.vocabulary)) > 0
                N_word_1_label_1[:, observation['label']] += words
                N_docs_label_1[0, observation['label']] += 1

        # Finalize dataset
        if not was_setup:
            self.dataset.clean('fit')

        # Setup count matrices for not-word, not-label, and not-word & not-label
        # The standard notation is count = P(U=u, C=c) * N
        N_word_1_label_0 = torch.sum(N_word_1_label_1, dim=1, keepdim=True) - N_word_1_label_1
        N_word_0_label_1 = N_docs_label_1 - N_word_1_label_1
        N_word_0_label_0 = torch.sum(N_word_0_label_1, dim=1, keepdim=True) - N_word_0_label_1

        N_label_1 = N_word_0_label_1 + N_word_1_label_1
        N_label_0 = N_word_0_label_0 + N_word_1_label_0
        N_word_1 = N_word_1_label_1 + N_word_1_label_0
        N_word_0 = N_word_0_label_1 + N_word_0_label_0

        # Compute the mutual information
        self.mutual_information = (
            (N_word_1_label_1 / N_docs) * (
                (torch.log2(N_docs) + torch.log2(N_word_1_label_1)) - (torch.log2(N_word_1) + torch.log2(N_label_1))
            ) +
            (N_word_1_label_0 / N_docs) * (
                (torch.log2(N_docs) + torch.log2(N_word_1_label_0)) - (torch.log2(N_word_1) + torch.log2(N_label_0))
            ) +
            (N_word_0_label_1 / N_docs) * (
                (torch.log2(N_docs) + torch.log2(N_word_0_label_1)) - (torch.log2(N_word_0) + torch.log2(N_label_1))
            ) +
            (N_word_0_label_0 / N_docs) * (
                (torch.log2(N_docs) + torch.log2(N_word_0_label_0)) - (torch.log2(N_word_0) + torch.log2(N_label_0))
            )
        )

    def forward(self, batch: SequenceBatch) -> torch.Tensor:
        importances = []
        for observation_i in range(batch.sentence.size(0)):
            mutual_information_for_label = self.mutual_information[:, batch.label[observation_i]]

            importances.append(
                torch.index_select(mutual_information_for_label, 0, batch.sentence[observation_i, :])
            )

        return torch.stack(importances)

class AttentionImportanceMeasure(ImportanceMeasureModule):
    def forward(self, batch: SequenceBatch) -> torch.Tensor:
        with torch.no_grad():
            _, alpha, _ = self.model(batch)
        return alpha

class GradientImportanceMeasure(ImportanceMeasureModule):
    def forward(self, batch: SequenceBatch) -> torch.Tensor:
        # Compute model
        y, _, embedding = self.model(batch)
        # Select correct label, as we would like gradient of y[correct_label] w.r.t. x
        yc = y[torch.arange(batch.label.numel(), device=self.device), batch.label]

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

class IntegratedGradientImportanceMeasure(ImportanceMeasureModule):
    riemann_samples: torch.Tensor

    def __init__(self, *args, riemann_samples=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.riemann_samples = torch.tensor(riemann_samples, device=self.device)

    def forward(self, batch: SequenceBatch) -> torch.Tensor:
        # Prepear a compact embedding matrix for doing sum(x * dy/dz @ W.T) efficently.
        embedding_matrix_compact = torch.index_select(
            self.model.embedding_matrix, 0, batch.sentence.view(-1)
        ).unsqueeze(-1)

        # Riemann approximation of the integral
        online_mean = torch.zeros_like(batch.sentence,
                                       dtype=self.model.embedding_matrix.dtype,
                                       device=self.device)
        for i in torch.arange(1, self.riemann_samples + 1, device=self.device):
            embedding_scale = i / self.riemann_samples
            y, _, embedding = self.model(batch, embedding_scale=embedding_scale)
            yc = y[torch.arange(batch.label.numel(), device=self.device), batch.label]
            yc_batch = yc.sum(dim=0)

            with torch.no_grad():
                yc_wrt_embedding, = torch.autograd.grad([yc_batch], (embedding, )) # (B, T, Z)
                if yc_wrt_embedding is None:
                    raise ValueError('Could not compute gradient')

                # This is a fast and memory-efficient version of sum(one_hot(x) * dy/dz @ W.T)
                # We can do this because x is one_hot, hence there is no need to
                # compute all the dy/dx = dy/dz @ W.T elements, where x = 0,
                # because they will anyway go away after sum.
                # In this context, the sum comes from the 2-norm. The mean
                # does not affect anything, as x remains the same for all
                # # Riemann steps.
                yc_wrt_x_compact = torch.bmm(
                    yc_wrt_embedding.view(
                        embedding_matrix_compact.shape[0], 1, embedding_matrix_compact.shape[1]
                    ), # (B * T, 1, Z)
                    embedding_matrix_compact, # (B * T, Z, 1)
                ).view_as(batch.sentence) # (B*T, 1, 1) -> (B, T)

                # Update the online mean (Knuth Algorithm), this is more memory
                # efficient that storing x_yc_wrt_x for each Riemann step.
                online_mean += (yc_wrt_x_compact - online_mean)/i

        # Abs is equivalent to 2-norm, because the naive sum is essentially
        # sqrt(0^2 + ... + 0^2 + y_wrt_x^2 + 0^2 + ... + 0^2) = abs(y_wrt_x)
        return torch.abs(online_mean)

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

class ImportanceMeasure:
    def __init__(self, model, dataset, importance_measure,
                 riemann_samples=50, use_gpu=False, num_workers=4, batch_size=None, seed=0,
                 caching=None, cachedir=None):
        if caching not in [None, 'use', 'build']:
            raise ValueError('caching argument must be either None, "use" or "build"')

        self._dataset = dataset
        self._np_rng = np.random.RandomState(seed)

        self._num_workers = num_workers
        self._batch_size = batch_size
        self._caching = caching
        self._cachedir = cachedir
        self._cachename = generate_experiment_id(dataset.name, seed,
                                                 importance_measure=importance_measure,
                                                 riemann_samples=riemann_samples)

        if importance_measure == 'random':
            self._use_gpu = False
            self._importance_measure_fn = RandomImportanceMeasure(model, dataset, use_gpu=self._use_gpu, rng=self._np_rng)
        elif importance_measure == 'mutual-information':
            self._use_gpu = use_gpu
            measure = MutualInformationImportanceMeasure(model, dataset, use_gpu=self._use_gpu, rng=self._np_rng)
            measure.precompute(batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False)
            self._importance_measure_fn = torch.jit.script(measure)
        elif importance_measure == 'attention':
            self._use_gpu = use_gpu
            self._importance_measure_fn = torch.jit.script(
                AttentionImportanceMeasure(model, dataset, use_gpu=self._use_gpu, rng=self._np_rng))
        elif importance_measure == 'gradient':
            self._use_gpu = use_gpu
            self._importance_measure_fn = torch.jit.script(
                GradientImportanceMeasure(model, dataset, use_gpu=self._use_gpu, rng=self._np_rng))
        elif importance_measure == 'integrated-gradient':
            self._use_gpu = use_gpu
            self._importance_measure_fn = torch.jit.script(
                IntegratedGradientImportanceMeasure(model, dataset, riemann_samples=riemann_samples,
                                                    use_gpu=self._use_gpu, rng=self._np_rng))
        else:
            raise ValueError(f'{importance_measure} is not supported')

    def evaluate(self, split):
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
