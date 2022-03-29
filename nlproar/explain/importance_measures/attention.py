
import torch

from ...dataset import SequenceBatch
from ._abstract import ImportanceMeasureModule

class AttentionImportanceMeasure(ImportanceMeasureModule):
    def forward(self, batch: SequenceBatch) -> torch.Tensor:
        with torch.no_grad():
            _, alpha, _ = self.model(batch)
        return alpha
