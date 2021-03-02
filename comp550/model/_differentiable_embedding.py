
import torch
import torch.nn as nn

class DifferentiableEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(*args, **kwargs)

    def forward(self, x):
        # To support gradient w.r.t. input, we must allow x to be a float-tensor.
        if torch.is_floating_point(x):
            h = torch.matmul(x, self.embedding.weight)
        else:
            h = self.embedding(x)

        return h
