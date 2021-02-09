
import torch
import torch.nn as nn

class DifferentiableEmbedding(nn.Embedding):
    def forward(self, x):
        # To support gradient w.r.t. input, we must allow x to be a float-tensor.
        if torch.is_floating_point(x):
            h = torch.matmul(x, self.weight)
        else:
            h = super().forward(x)

        return h
