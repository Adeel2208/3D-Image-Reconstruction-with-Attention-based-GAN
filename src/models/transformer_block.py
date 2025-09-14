import torch
import torch.nn as nn

class MultiScaleFeatureAggregation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 4
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        valid_groups = 1
        for g in range(1, min(in_channels, mid_channels) + 1):
            if in_channels % g == 0 and mid_channels % g == 0:
                valid_groups = g
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, groups=valid_groups),
            nn.Conv2d(mid_channels, mid_channels, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)
