import torch
import torch.nn as nn
import torch.nn.functional as F
from opencd.registry import MODELS

@MODELS.register_module()
class C_CosineSimilarityLoss(nn.Module):
    """Cosine Similarity Loss with Contrastive and Masking Operations"""

    def __init__(self, loss_weight=1.0, loss_name='cosine_similarity_loss', epsilon=1e-5):
        super().__init__()
        self.weight = loss_weight
        self._loss_name = loss_name
        self.epsilon = epsilon
        self.balance_weight = 0.4

    def forward(self, pred, target, mask):
        # Apply masking
        mask = torch.argmax(mask, 1).unsqueeze(1)
        p_pred = (1 - mask) * pred
        p_target = (1 - mask) * target
        n_pred = mask * pred
        n_target = mask * target

        # Compute loss with balance weight
        return self.weight * (self.balance_weight * self.cosine_loss(p_pred, p_target) - 
                              (1 - self.balance_weight) * self.cosine_loss(n_pred, n_target))

    def cosine_loss(self, pred, target):
        # Normalize the predictions and targets
        # pred = self.extract_non_zero_features(pred)
        # target = self.extract_non_zero_features(target)
        pred_norm = F.normalize(pred, p=2, dim=1)
        target_norm = F.normalize(target, p=2, dim=1)

        # Cosine similarity
        cosine_sim = torch.sum(pred_norm * target_norm, dim=1)

        # Cosine similarity loss
        return 1 - torch.mean(cosine_sim)
    def extract_non_zero_features(self, features):
        # 计算每个特征向量的范数
        norms = torch.norm(features, dim=(2, 3))
        # 创建一个布尔掩码，标识非零向量
        mask = norms != 0
        # 使用掩码来压缩特征向量，只保留非零向量
        non_zero_features = features[mask]
        # 重新排列输出以匹配所需的维度 (B, N, m)
        non_zero_features = non_zero_features.view(features.size(0), features.size(1), -1)
        return non_zero_features
    
    @property
    def loss_name(self):
        """Loss Name."""
        return self._loss_name
