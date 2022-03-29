
import torch

from ...dataset import SequenceBatch
from ._abstract import ImportanceMeasureModule

class GradientImportanceMeasure(ImportanceMeasureModule):
    def forward(self, batch: SequenceBatch) -> torch.Tensor:
        # Compute model
        y, _, embedding = self.model(batch)
        # Select correct label, as we would like gradient of y[correct_label] w.r.t. x
        # Note, we want an explanation for the correct label, as removing the tokens
        # that are relevant for making a wrong prediction, would help the performance
        # of the model.
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
