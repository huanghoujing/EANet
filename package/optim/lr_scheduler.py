from torch.optim.lr_scheduler import _LRScheduler

"""The get_lr method of _LRScheduler can be interpreted as a math function that 
takes parameter `self.last_epoch` and returns the current lr.
Note that `epoch` in pytorch lr scheduler is only local naming, which in fact means
one time of [calling **LR.step()] instead of [going over the whole dataset].
"""


class WarmupLR(_LRScheduler):
    def __init__(self, optimizer, epochs, last_epoch=-1):
        super(WarmupLR, self).__init__(optimizer, last_epoch)
        self.epochs = epochs
        self.final_lrs = [group['warmup_final_lr'] for group in optimizer.param_groups]

    def get_lr(self):
        """A linear function, increasing from base_lr to final_lr."""
        return [base_lr + 1. * self.last_epoch * (final_lr - base_lr) / self.epochs
                for base_lr, final_lr in zip(self.base_lrs, self.final_lrs)]
