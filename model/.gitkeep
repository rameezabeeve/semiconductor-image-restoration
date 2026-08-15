import torch
import torch.nn as nn


class RestorationModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 1, 3, padding=1)
        )

        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.upsample(x)
        x = self.output_activation(x)
        return x
