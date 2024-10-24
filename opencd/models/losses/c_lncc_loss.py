import torch
import torch.nn as nn
import torch.nn.functional as F
from opencd.registry import MODELS

@MODELS.register_module()
class C_LNCCLoss(nn.Module):
    """Local Normalized Cross-Correlation Loss"""

    def __init__(self, window_size=9, loss_weight=1.0, loss_name='lncc_loss', epsilon=1e-5):
        super().__init__()
        self.window_size = window_size
        self.weight = loss_weight
        self._loss_name = loss_name
        self.epsilon = epsilon
        self.balance_weight = 0.2
    def forward(self, pred, target, mask):
        pred1 = pred
        pred2 = target
        mask = torch.argmax(mask,1).unsqueeze(1)
        p_pred1 = (1-mask)*pred1
        p_pred2 = (1-mask)*pred2
        n_pred1 = (mask)*pred1
        n_pred2 = (mask)*pred2  
        return self.weight * (self.balance_weight*self.lncc_loss(p_pred1,p_pred2)-(1-self.balance_weight)*self.lncc_loss(n_pred1,n_pred2))

    @staticmethod
    def compute_local_sum(x, window_size):
        padding = window_size // 2
        sum_x = F.conv2d(x, torch.ones((1, 1, window_size, window_size), dtype=x.dtype, device=x.device), padding=padding)
        return sum_x

    def lncc_loss(self, pred, target):
        pred_mean = self.compute_local_sum(pred, self.window_size) / (self.window_size ** 2)
        target_mean = self.compute_local_sum(target, self.window_size) / (self.window_size ** 2)

        pred_variance = torch.clamp(self.compute_local_sum(pred ** 2, self.window_size) - pred_mean ** 2, min=0)
        target_variance = torch.clamp(self.compute_local_sum(target ** 2, self.window_size) - target_mean ** 2, min=0)

        covariance = self.compute_local_sum(pred * target, self.window_size) - pred_mean * target_mean

        lncc = covariance / (torch.sqrt(pred_variance + self.epsilon) * torch.sqrt(target_variance + self.epsilon))

        return 1-torch.mean(lncc)

    @property
    def loss_name(self):
        """Loss Name."""
        return self._loss_name



# @MODELS.register_module()
# class C_LNCCLoss(nn.Module):
#     """Local Normalized Cross-Correlation Loss"""

#     def __init__(self, window_size=9, loss_weight=1.0, loss_name='lncc_loss'):
#         super().__init__()
#         self.window_size = window_size
#         self.weight = loss_weight
#         self._loss_name = loss_name
#         self.balance_weight = 0.2

#     def forward(self, pred, target, mask):

#         pred1 = pred.mean(dim=1, keepdim=True)
#         pred2 = target.mean(dim=1, keepdim=True)
#         mask = torch.argmax(mask,1).unsqueeze(1)
#         p_pred1 = (1-mask)*pred1
#         p_pred2 = (1-mask)*pred2
#         n_pred1 = (mask)*pred1
#         n_pred2 = (mask)*pred2  
#         return self.weight * (self.balance_weight*self.lncc_loss(p_pred1,p_pred2)+(1-self.balance_weight)*self.lncc_loss(n_pred1,n_pred2))
        

#     @staticmethod
#     def compute_local_sum(x, window_size):
#         padding = window_size // 2
#         sum_x = F.conv2d(x, torch.ones((1, 1, window_size, window_size), dtype=x.dtype, device=x.device), padding=padding)
#         return sum_x

#     def lncc_loss(self, pred, target):
#         pred_mean = self.compute_local_sum(pred, self.window_size) / (self.window_size ** 2)
#         target_mean = self.compute_local_sum(target, self.window_size) / (self.window_size ** 2)

#         pred_variance = self.compute_local_sum(pred ** 2, self.window_size) - pred_mean ** 2
#         target_variance = self.compute_local_sum(target ** 2, self.window_size) - target_mean ** 2

#         covariance = self.compute_local_sum(pred * target, self.window_size) - pred_mean * target_mean

#         lncc = covariance / (torch.sqrt(pred_variance) * torch.sqrt(target_variance) + 1e-5)
#         print(torch.mean(lncc))
#         return 1-torch.mean(lncc)

#     @property
#     def loss_name(self):
#         """Loss Name."""
#         return self._loss_name
