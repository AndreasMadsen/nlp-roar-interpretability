
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class ExampleDataset(pl.LightningDataModule):
      def __init__(self, batch_size=32, validation_ratio=0.1, seed=0):
          super().__init__()
          self.batch_size = batch_size

      def prepare_data(self):
          pass

      def setup(self, stage):
          pass

      def train_dataloader(self):
          return DataLoader(self._train, batch_size=self.batch_size)

      def val_dataloader(self):
          return DataLoader(self._val, batch_size=self.batch_size)

      def test_dataloader(self):
          return DataLoader(self._test, batch_size=self.batch_size)
