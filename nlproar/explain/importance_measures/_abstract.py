
import torch
import torch.nn as nn

class ImportanceMeasureModule(nn.Module):
    def __init__(self, model, dataset, use_gpu, rng):
        super().__init__()
        self.model = model.cuda() if use_gpu else model
        self.model.flatten_parameters()
        self.dataset = dataset
        self.device = torch.device('cuda' if use_gpu else 'cpu')
        self.rng = rng
