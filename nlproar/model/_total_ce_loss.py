import torch
import torch.nn.functional as F
import torchmetrics

class TotalCrossEntropyLoss(torchmetrics.Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.sum += F.cross_entropy(preds, target, reduction='sum')
        self.count += torch.tensor(target.numel(), device=target.device)

    def compute(self) -> torch.Tensor:
        return self.sum.float() / self.count
