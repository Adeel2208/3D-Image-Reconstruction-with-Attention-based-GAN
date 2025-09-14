import torch
import torch.nn as nn

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super(PatchDiscriminator, self).__init__()
        layers = []
        for idx, feature in enumerate(features):
            if idx == 0:
                layers.append(
                    nn.utils.spectral_norm(
                        nn.Conv2d(in_channels, feature, kernel_size=4, stride=2, padding=1)
                    )
                )
            else:
                layers.append(
                    nn.utils.spectral_norm(
                        nn.Conv2d(features[idx-1], feature, kernel_size=4, stride=2, padding=1)
                    )
                )
            if idx != len(features) - 1:
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                layers.append(nn.BatchNorm2d(feature))
            else:
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                layers.append(nn.Conv2d(feature, 1, kernel_size=4, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
