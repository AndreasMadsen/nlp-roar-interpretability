
from tqdm import tqdm
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class ROARDataset(pl.LightningDataModule):
    """Loads a dataset with masking for ROAR."""

    def __init__(
        self,
        model,
        base_dataset,
        k=1,
        random_masking=False,
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
            random_masking (bool): Whether to mask tokens randomly or not.
        """
        super().__init__()
        self._model = model
        self._base_dataset = base_dataset
        self._seed = seed
        self._num_workers = num_workers
        self._batch_size = batch_size
        self._k = k
        self._random_masking = random_masking
        self.tokenizer = base_dataset.tokenizer

    def _mask_batch(self, batch):
        h3, alpha = self._model(batch)

        for idx, (length, mask) in enumerate(zip(batch["length"], batch["mask"])):
            n_tokens = min(self._k, length - 2)
            if self._random_masking:
                # Make sure only tokens that can be attended to are masked
                attended_indices = torch.where(mask[:length])[0]
                # Shuffle attended_indices
                attended_indices = attended_indices[torch.randperm(len(attended_indices))]
                # Select n_tokens
                token_indices = attended_indices[:n_tokens]
            else:
                # Ensure that alpha is non-zero for attended tokens
                #  and zero for non-attended tokens
                alpha = torch.where(mask,
                                    alpha + torch.tensor(1e-6, dtype=alpha.dtype),
                                    torch.tensor(0, dtype=alpha.dtype))
                _, token_indices = alpha[idx, :length].topk(k=n_tokens, dim=-1)

            # Mask tokens for ROAR
            batch["sentence"][idx, token_indices] = self.tokenizer.mask_token_id

        return self._base_dataset.uncollate(batch)

    def _process_data(self, data, name):
        outputs = []
        for batch in tqdm(data, desc=f'Building {name} dataset', leave=False):
            outputs.extend(self._mask_batch(batch))
        return outputs

    def setup(self, stage):
        self._base_dataset.setup(stage)
        if stage == "fit":
            self._train = self._process_data(self._base_dataset.train_dataloader(), 'train')
            self._val = self._process_data(self._base_dataset.val_dataloader(), 'val')
        elif stage == 'test':
            self._test = self._process_data(self._base_dataset.test_dataloader(), 'test')
        else:
            raise ValueError(f'unexpected setup stage: {stage}')

    def collate(self, observations):
        return self._base_dataset.collate(observations)

    def uncollate(self, observations):
        return self._base_dataset.uncollate(observations)

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
