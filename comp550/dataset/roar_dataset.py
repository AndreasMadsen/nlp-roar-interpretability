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
        batch_size=32,
        seed=0,
        num_workers=4,
    ):
        super().__init__()
        """
        Args:
            model: The model to use to determine which tokens to mask.
            base_dataset: The dataset to apply masking to.
            k (float): The proportion of tokens to mask for each instance in the
                dataset.
            do_random_masking (bool): Whether to mask tokens randomly or not.
        """
        if k < 0 or k > 1:
            raise ValueError("Invalid value for k: %s." % k)

        self._model = model
        self._base_dataset = base_dataset
        self.k = k
        self.do_random_masking = do_random_masking
        self._batch_size = batch_size
        self._seed = seed
        self._num_workers = num_workers
        self.tokenizer = base_dataset.tokenizer

    def _get_masked_instance(self, instance, token_indices):
        # Mask tokens for ROAR
        instance["sentence"][token_indices] = self._base_dataset.tokenizer.mask_token_id
        for k in instance:
            # Remove padding
            if k in ["sentence", "mask"]:
                instance[k] = instance[k][: instance["length"]]
            elif k in ["hypothesis", "hypothesis_mask"]:
                instance[k] = instance[k][: instance["hypothesis_length"]]

        return instance

    def _mask_batch(self, batch):
        outputs = []

        h3, alpha = self._model(batch)

        for idx, len_ in enumerate(batch["length"]):
            n_tokens = math.ceil(self.k * len_)

            if self.do_random_masking:
                token_indices = torch.randperm(len(alpha[idx]))[:n_tokens]
            else:
                _, token_indices = alpha[idx].topk(k=n_tokens, dim=-1)

            instance = self._get_masked_instance(
                instance={k: v[idx] for k, v in batch.items()},
                token_indices=token_indices,
            )
            outputs.append(instance)

        return outputs

    def _process_data(self, data):
        dataloader = DataLoader(
            data,
            batch_size=self._batch_size,
            collate_fn=self._base_dataset._collate,
            num_workers=self._num_workers,
        )

        outputs = []
        for batch in tqdm(dataloader):
            outputs.extend(self._mask_batch(batch))
        return outputs

    def setup(self, stage):
        if stage not in ["fit", "test"]:
            raise ValueError("Unexpected setup stage: %s." % stage)

        self._base_dataset.setup(stage)
        if stage == "fit":
            self._train = self._process_data(self._base_dataset._train)
            self._val = self._process_data(self._base_dataset._val)
        else:
            self._test = self._process_data(self._base_dataset._test)

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
