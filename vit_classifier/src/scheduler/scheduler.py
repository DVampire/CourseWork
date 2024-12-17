from transformers import get_cosine_schedule_with_warmup

from src.registry import SCHEDULER


@SCHEDULER.register_module(force=True)
class CosineWithWarmupScheduler:
    def __init__(
        self,
        optimizer,
        num_warmup_steps,
        num_training_steps,
        num_cycles=0.5,
        last_epoch=-1,
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

        self._scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            last_epoch=last_epoch,
        )

    def __str__(self):
        return f'CosineWithWarmupScheduler(num_warmup_steps={self.num_warmup_steps}, num_training_steps={self.num_training_steps})'

    def __getattr__(self, name):
        return getattr(self._scheduler, name)
