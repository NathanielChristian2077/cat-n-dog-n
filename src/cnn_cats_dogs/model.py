"""CNN architecture built manually from PyTorch layers, with no pretrained backbone."""

from __future__ import annotations

import torch
from torch import nn


class ConvNormAct(nn.Sequential):
    """Convolution -> batch normalization -> ReLU building block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ScratchCNN(nn.Module):
    """A compact CNN intentionally authored for the assignment.

    Architecture summary:
      RGB 224x224
        -> [Conv 3->32, Conv 32->32, MaxPool]
        -> [Conv 32->64, Conv 64->64, MaxPool]
        -> [Conv 64->128, Conv 128->128, MaxPool]
        -> AdaptiveAvgPool -> Fully connected 128->128->1 logit.

    It has six convolutional layers, three pooling layers, and fully connected
    output layers. No torchvision architecture or pretrained weights are imported.
    """

    def __init__(self, dropout: float = 0.35) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvNormAct(3, 32),
            ConvNormAct(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.10),
            ConvNormAct(32, 64),
            ConvNormAct(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.15),
            ConvNormAct(64, 128),
            ConvNormAct(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.20),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
