import torch
import torch.nn as nn

class FrequencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
        target_fft = torch.fft.rfft2(target, dim=(-2, -1))
        
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        B, C, H, W = pred.shape
        freq_weight = torch.ones_like(pred_mag)
        freq_weight[:, :, H//4:3*H//4, :W//4] = 0.5
        
        return nn.functional.l1_loss(pred_mag * freq_weight, target_mag * freq_weight)
