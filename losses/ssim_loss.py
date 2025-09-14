import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleSSIMLoss(nn.Module):
    def __init__(self, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        
    def forward(self, pred, target):
        loss = 0
        for scale in range(self.num_scales):
            if scale > 0:
                pred = F.avg_pool2d(pred, 2)
                target = F.avg_pool2d(target, 2)
            
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2
            
            mu_pred = F.avg_pool2d(pred, 3, 1, 1)
            mu_target = F.avg_pool2d(target, 3, 1, 1)
            
            mu_pred_sq = mu_pred ** 2
            mu_target_sq = mu_target ** 2
            mu_pred_target = mu_pred * mu_target
            
            sigma_pred_sq = F.avg_pool2d(pred ** 2, 3, 1, 1) - mu_pred_sq
            sigma_target_sq = F.avg_pool2d(target ** 2, 3, 1, 1) - mu_target_sq
            sigma_pred_target = F.avg_pool2d(pred * target, 3, 1, 1) - mu_pred_target
            
            ssim = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
                   ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
            
            loss += (1 - ssim.mean()) * (2 ** (self.num_scales - scale - 1))
            
        return loss / (2 ** self.num_scales - 1)
