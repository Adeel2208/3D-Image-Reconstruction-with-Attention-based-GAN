```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import WindowAttention, window_partition, window_reverse
from .frequency_enhancement import FrequencyEnhancementModule
from .multi_scale_aggregation import MultiScaleFeatureAggregation
from .transformer_block import HybridTransformerBlock

class HybridVisionTransformerUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512], 
                 window_sizes=[32, 16, 8, 8], num_heads=[2, 4, 8, 8], depths=[2, 2, 2, 2]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.frequency_enhancers = nn.ModuleList()
        self.transformer_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        for i, feature in enumerate(features):
            conv_block = nn.Sequential(
                MultiScaleFeatureAggregation(in_channels, feature),
                nn.Conv2d(feature, feature, 3, padding=1),
                nn.BatchNorm2d(feature),
                nn.ReLU(inplace=True)
            )
            self.encoder.append(conv_block)
            
            if i >= 2:
                self.frequency_enhancers.append(FrequencyEnhancementModule(feature))
            else:
                self.frequency_enhancers.append(nn.Identity())
            
            if i >= 2:
                transformer_list = nn.ModuleList([
                    HybridTransformerBlock(feature, num_heads[i], window_sizes[i]) 
                    for _ in range(depths[i])
                ])
                self.transformer_blocks.append(transformer_list)
            else:
                self.transformer_blocks.append(nn.ModuleList([nn.Identity() for _ in range(depths[i])]))
            
            in_channels = feature
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1]*2, 3, padding=1),
            nn.BatchNorm2d(features[-1]*2),
            nn.ReLU(inplace=True),
            FrequencyEnhancementModule(features[-1]*2)
        )
        
        self.upconvs = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.skip_attns = nn.ModuleList()
        
        reversed_features = features[::-1]
        for i, feature in enumerate(reversed_features):
            self.upconvs.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(
                nn.Sequential(
                    nn.Conv2d(feature*2, feature, 3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, 3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True)
                )
            )
            self.skip_attns.append(
                nn.Sequential(
                    nn.Conv2d(feature, 1, 1),
                    nn.Sigmoid()
                )
            )
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], features[0]//2, 3, padding=1),
            nn.BatchNorm2d(features[0]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0]//2, out_channels, 1)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        skip_connections = []
        
        for i, (conv, freq_enh, trans_blocks) in enumerate(
            zip(self.encoder, self.frequency_enhancers, self.transformer_blocks)):
            
            x = conv(x)
            x = freq_enh(x)
            
            if i >= 2 and len(trans_blocks) > 0 and not isinstance(trans_blocks[0], nn.Identity):
                B, C, H, W = x.shape
                x_flat = x.flatten(2).transpose(1, 2)
                for trans in trans_blocks:
                    x_flat = trans(x_flat, H, W)
                x = x_flat.transpose(1, 2).reshape(B, C, H, W)
            
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        
        skip_connections = skip_connections[::-1]
        for idx in range(len(self.upconvs)):
            x = self.upconvs[idx](x)
            skip = skip_connections[idx]
            skip_attn = self.skip_attns[idx](skip)
            skip = skip * skip_attn
            x = torch.cat((skip, x), dim=1)
            x = self.decoder[idx](x)
        
        x = self.final_conv(x)
        x = self.sigmoid(x)
        return x
```