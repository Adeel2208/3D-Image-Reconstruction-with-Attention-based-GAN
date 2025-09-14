import torch
import torch.nn as nn
from torchvision import models

class PerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.layers = nn.Sequential(*[vgg[i] for i in range(16)])
        for param in self.layers.parameters():
            param.requires_grad = False
        self.resize = resize
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, input, target):
        if self.resize:
            input = nn.functional.interpolate(input, size=(224, 224), mode='bilinear', align_corners=False)
            target = nn.functional.interpolate(target, size=(224, 224), mode='bilinear', align_corners=False)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        input_features = self.layers(input)
        target_features = self.layers(target)
        return nn.functional.l1_loss(input_features, target_features)