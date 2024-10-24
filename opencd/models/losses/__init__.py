from .bcl_loss import BCLLoss
from .contrastive_Loss import ContrastiveLoss
from .mse_loss import MSELoss
from .c_ssim import C_SSIMLoss
from .c_lncc_loss import C_LNCCLoss
from .c_cosine import C_CosineSimilarityLoss
__all__ = ['BCLLoss','ContrastiveLoss','MSELoss','C_SSIMLoss','C_LNCCLoss',"C_CosineSimilarityLoss"]