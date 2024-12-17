import torch.nn as nn

from src.registry import CRITERION


@CRITERION.register_module(force=True)
class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, reduction='mean', label_smoothing=0.1):
        super().__init__(weight=weight, reduction=reduction, label_smoothing=label_smoothing)
