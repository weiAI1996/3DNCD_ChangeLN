import torch
import torch.nn as nn
import torch.nn.functional as F
from opencd.registry import MODELS

def gaussian_window(size, sigma):
    coords = torch.arange(size, dtype=torch.float32)
    coords -= size // 2

    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    return g.view(1, 1, 1, -1)

def ssim(pred, target, window_size=11, window_sigma=1.5, size_average=True):
    channel = pred.size(1)
    window = gaussian_window(window_size, window_sigma).to(pred.device).repeat(channel, 1, 1, 1)

    mu1 = F.conv2d(pred, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size//2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, window_sigma=1.5, size_average=True, loss_weight=1.0, loss_name='ssim_loss'):
        super().__init__()
        self.window_size = window_size
        self.window_sigma = window_sigma
        self.size_average = size_average
        self.weight = loss_weight
        self._loss_name = loss_name

    def forward(self, pred, target):
        return self.weight * (1 - ssim(pred, target, self.window_size, self.window_sigma, self.size_average))

    @property
    def loss_name(self):
        return self._loss_name

@MODELS.register_module()
class C_SSIMLoss(nn.Module):
    def __init__(self, loss_weight=1.0, loss_name='c_ssim_loss'):
        super().__init__()
        self.ssim = SSIMLoss()
        self.weight = loss_weight
        self._loss_name = loss_name
        self.balance_weight = 0.2

    def forward(self, pred, target, mask):

        pred1 = pred.mean(dim=1, keepdim=True)
        pred2 = target.mean(dim=1, keepdim=True)
        mask = torch.argmax(mask,1).unsqueeze(1)
        p_pred1 = (1-mask)*pred1
        p_pred2 = (1-mask)*pred2
        n_pred1 = (mask)*pred1
        n_pred2 = (mask)*pred2        
        return self.weight * (self.balance_weight*self.ssim(p_pred1,p_pred2)-(1-self.balance_weight)*self.ssim(n_pred1,n_pred2))

    @property
    def loss_name(self):
        return self._loss_name
