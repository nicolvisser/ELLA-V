"""
This module implements `LinearRampCosineDecayScheduler`, a custom PyTorch 
learning rate scheduler that linearly increases the learning rate, then 
applies cosine decay, and finally maintains a constant rate.
"""

import math

import torch.optim as optim
from torch.optim.optimizer import Optimizer


class LinearRampCosineDecayScheduler(optim.lr_scheduler._LRScheduler):
    """
    Custom learning rate scheduler that increases linearly for n_linear_steps,
    then decays with cosine annealing for n_decay_steps,
    then stays at lr_final for the remaining steps.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        n_linear_steps (int): Number of steps for linear increase.
        n_decay_steps (int): Number of steps for cosine decay.
        lr_init (float, optional): Initial learning rate. Default is 0.
        lr_max (float, optional): Maximum learning rate. Default is 1e-5.
        lr_final (float, optional): Final learning rate. Default is 1e-6.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        n_linear_steps: int,
        n_decay_steps: int,
        lr_init: float,
        lr_max: float,
        lr_final: float,
    ):
        self.n_linear_steps = n_linear_steps
        self.n_decay_steps = n_decay_steps

        self.lr_init = lr_init
        self.lr_max = lr_max
        self.lr_final = lr_final

        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        current_step = self.last_epoch

        if current_step <= self.n_linear_steps:
            lr = self.lr_init + (self.lr_max - self.lr_init) * current_step / (
                self.n_linear_steps
            )
        elif current_step <= self.n_linear_steps + self.n_decay_steps:
            lr = (
                0.5
                * math.cos(
                    (current_step - self.n_linear_steps)
                    / (self.n_decay_steps)
                    * math.pi
                )
                + 0.5
            ) * (self.lr_max - self.lr_final) + self.lr_final
        else:
            lr = self.lr_final
        return [lr for _ in self.base_lrs]
