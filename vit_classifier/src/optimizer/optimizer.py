from torch.optim import AdamW

from src.registry import OPTIMIZER

OPTIMIZER.register_module(name='AdamW', module=AdamW)
