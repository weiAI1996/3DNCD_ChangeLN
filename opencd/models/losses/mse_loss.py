import torch
import torch.nn as nn
from opencd.registry import MODELS
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def mse_loss(pred, target):
    return torch.mean((pred - target) ** 2)

@MODELS.register_module()
class MSELoss(nn.Module):
    """Mean Squared Error Loss"""

    def __init__(self, loss_weight=1.0, loss_name='mse_loss'):
        super().__init__()
        self.weight = loss_weight
        self._loss_name = loss_name

    def forward(self, pred, target):
        return self.weight * mse_loss(pred, target)

    @property
    def loss_name(self):
        """Loss Name."""
        return self._loss_name
