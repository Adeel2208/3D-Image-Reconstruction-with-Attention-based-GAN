import torch
import torch.nn as nn

class FrequencyEnhancementModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.conv_high = nn.Conv2d(channels, channels, 1)
        self.conv_low = nn.Conv2d(channels, channels, 1)
        self.fusion = nn.Conv2d(channels * 2, channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x_fft = torch.fft.rfft2(x, dim=(-2, -1))
        
        freq_h = torch.fft.fftfreq(H, device=x.device).reshape(H, 1)
        freq_w = torch.fft.rfftfreq(W, device=x.device).reshape(1, W // 2 + 1)
        freq_mag = torch.sqrt(freq_h ** 2 + freq_w ** 2)
        
        high_mask = (freq_mag > 0.25).float().unsqueeze(0).unsqueeze(0)
        low_mask = 1 - high_mask
        
        x_high = torch.fft.irfft2(x_fft * high_mask, s=(H, W), dim=(-2, -1))
        x_low = torch.fft.irfft2(x_fft * low_mask, s=(H, W), dim=(-2, -1))
        
        x_high = self.conv_high(x_high)
        x_low = self.conv_low(x_low)
        
        x_fused = torch.cat([x_high, x_low], dim=1)
        x_out = self.fusion(x_fused)
        
        return x_out + x
