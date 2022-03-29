
import torch

from ...dataset import SequenceBatch
from ._abstract import ImportanceMeasureModule

class InputTimesGradientImportanceMeasure(ImportanceMeasureModule):
    def forward(self, batch: SequenceBatch) -> torch.Tensor:
        # Prepear a compact embedding matrix for doing sum(x * dy/dz @ W.T) efficently.
        embedding_matrix_compact = torch.index_select(
            self.model.embedding_matrix, 0, batch.sentence.view(-1)
        ).unsqueeze(-1)  # (B * T, Z, 1)

        y, _, embedding = self.model(batch)
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

        # Abs is equivalent to 2-norm, because the naive sum is essentially
        # sqrt(0^2 + ... + 0^2 + y_wrt_x^2 + 0^2 + ... + 0^2) = abs(y_wrt_x)
        return torch.abs(yc_wrt_x_compact)
