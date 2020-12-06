import math

from tqdm import tqdm
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader


class ROARDataset(pl.LightningDataModule):
    """Loads a dataset with masking for ROAR."""

    def __init__(
        self,
        model,
        base_dataset,
        k,
        do_random_masking=False,
        batch_size=128,
        seed=0,
        num_workers=4,
    ):
        """
        Args:
            model: The model to use to determine which tokens to mask.
            base_dataset: The dataset to apply masking to.
            k (int): The number of tokens to mask for each instance in the
                dataset.
            do_random_masking (bool): Whether to mask tokens randomly or not.
        """
        super().__init__()
        self._model = model
        self._base_dataset = base_dataset
        self.k = k
        self.do_random_masking = do_random_masking
        self._batch_size = batch_size
        self._seed = seed
        self._num_workers = num_workers
        self.tokenizer = base_dataset.tokenizer

    def _mask_batch(self, batch):
        h3, alpha = self._model(batch)

        for idx, len_ in enumerate(batch["length"]):
            attended_indices = torch.nonzero(batch["mask"][idx]).squeeze()

            n_tokens = min(self.k, len_)
            if self.do_random_masking:
                token_indices = attended_indices[torch.randperm(len(attended_indices))][
                    :n_tokens
                ]
            else:
                _, token_indices = alpha[idx, attended_indices].topk(k=n_tokens, dim=-1)

            # Mask tokens for ROAR
            batch["sentence"][
                idx, token_indices
            ] = self._base_dataset.tokenizer.mask_token_id

        return self._base_dataset.uncollate(batch)

    def _process_data(self, data):
        outputs = []
        for batch in tqdm(data):
            outputs.extend(self._mask_batch(batch))
        return outputs

    def setup(self, stage):
        if stage not in ["fit", "test"]:
            raise ValueError("Unexpected setup stage: %s." % stage)

        self._base_dataset.setup(stage)
        if stage == "fit":
            self._train = self._process_data(self._base_dataset.train_dataloader())
            self._val = self._process_data(self._base_dataset.val_dataloader())
        else:
            self._test = self._process_data(self._base_dataset.test_dataloader())

    def _collate(self, observations):
        return self._base_dataset._collate(observations)

    def train_dataloader(self):
        return DataLoader(
            self._train,
            batch_size=self._batch_size,
            collate_fn=self._collate,
            num_workers=self._num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val,
            batch_size=self._batch_size,
            collate_fn=self._collate,
            num_workers=self._num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self._test,
            batch_size=self._batch_size,
            collate_fn=self._collate,
            num_workers=self._num_workers,
        )
