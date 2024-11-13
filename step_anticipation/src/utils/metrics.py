from typing import Any, Optional

import torch
from torchmetrics import Metric


class MultiHotAccuracy(Metric):
    is_differentiable = True
    higher_is_better = True
    full_state_update = True

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0, dtype=torch.long))
        self.add_state("total", default=torch.tensor(0, dtype=torch.long))

        self.threshold = threshold

    @torch.inference_mode()
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        # Convert to binary
        preds = (preds > self.threshold).int()
        target = target.int()

        # # ! we only consider the case where the target is 1
        # condition = ((target == 1) & (preds == target)).int()

        # ! we consider the case in which the whole vector is correct
        # ? maybe is too harsh
        preds = preds.view(-1, preds.shape[-1])
        target = target.view(-1, target.shape[-1])
        condition = preds == target

        self.correct = self.correct + torch.sum(torch.prod(condition, dim=-1))
        self.total = self.total + target.shape[0]

    @torch.inference_mode()
    def compute(self):
        return self.correct / self.total
