
import torch

from ...dataset import SequenceBatch
from ._abstract import ImportanceMeasureModule

class RandomImportanceMeasure(ImportanceMeasureModule):
    def forward(self, batch: SequenceBatch) -> torch.Tensor:
        return torch.tensor(self.rng.rand(*batch.sentence.shape))
