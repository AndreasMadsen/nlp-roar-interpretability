from tqdm import tqdm
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader


class ROARDataset(pl.LightningDataModule):
    """Loads a dataset with masking for ROAR.

    Attributes:
        k (int): The number of tokens to mask per instance.

    Notes:
        * Assumes that `base_dataset` has already been loaded and setup.
        * Assumes that `model` has already been loaded.
    """

    def __init__(
        self, cachedir, model, base_dataset, k, batch_size=32, seed=0, num_workers=4
    ):
        super().__init__()
        self._cachedir = cachedir
        self._model = model
        self._base_dataset = base_dataset
        self.k = k
        self._batch_size = batch_size
        self._seed = seed
        self._num_workers = num_workers
        self.tokenizer = base_dataset.tokenizer

    def _get_masked_instances(self, batch, top_k_token_indices):
        outputs = []
        for idx, token_indices in enumerate(top_k_token_indices):
            # Mask tokens for ROAR
            batch["sentence"][idx][
                token_indices
            ] = self._base_dataset.tokenizer.mask_token_id

            instance = {}
            for k in batch:
                if k in ["sentence", "mask"]:
                    instance[k] = batch[k][idx, :batch["length"][idx]]
                else:
                    instance[k] = batch[k][idx]
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
            h3, alpha = self._model(batch)
            _, top_k_token_indices = alpha.topk(k=self.k, dim=-1)
            outputs.extend(self._get_masked_instances(batch, top_k_token_indices))

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
