
import torch

from ...dataset import SequenceBatch
from ._abstract import ImportanceMeasureModule

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
