"""CNN architecture built manually from PyTorch layers, with no pretrained backbone."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

OutputMode = Literal["sigmoid1", "softmax2"]


class ConvNormAct(nn.Sequential):
    """Convolution -> batch normalization -> ReLU building block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ScratchCNN(nn.Module):
    """A scratch CNN with a fixed visual extractor and a configurable MLP head.

    The convolutional backbone is deliberately held constant between experiments.
    That lets the Stage 1 winners be compared fairly in the only place where a
    direct structural correspondence exists: the dense decision head.

    ``sigmoid1`` returns one raw logit and is trained with BCEWithLogitsLoss.
    ``softmax2`` returns two raw logits and is trained with CrossEntropyLoss.
    Neither activation is placed inside the model, because both PyTorch losses
    expect unnormalised logits for numerical stability.
    """

    def __init__(
        self,
        *,
        hidden_layers: tuple[int, ...] = (32, 64, 512),
        output_mode: OutputMode = "softmax2",
        classifier_dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if output_mode not in {"sigmoid1", "softmax2"}:
            raise ValueError("output_mode precisa ser 'sigmoid1' ou 'softmax2'.")
        if not hidden_layers or any(width < 1 for width in hidden_layers):
            raise ValueError("hidden_layers precisa conter larguras positivas.")
        if not 0.0 <= classifier_dropout < 1.0:
            raise ValueError("classifier_dropout precisa estar no intervalo [0, 1).")

        self.hidden_layers = hidden_layers
        self.output_mode = output_mode
        self.features = nn.Sequential(
            ConvNormAct(3, 32),
            ConvNormAct(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvNormAct(32, 64),
            ConvNormAct(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvNormAct(64, 128),
            ConvNormAct(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = self._make_classifier(
            hidden_layers=hidden_layers,
            output_mode=output_mode,
            classifier_dropout=classifier_dropout,
        )
        self._initialize_weights()

    @staticmethod
    def _make_classifier(
        *,
        hidden_layers: tuple[int, ...],
        output_mode: OutputMode,
        classifier_dropout: float,
    ) -> nn.Sequential:
        layers: list[nn.Module] = [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()]
        in_features = 128
        for index, width in enumerate(hidden_layers):
            layers.extend((nn.Linear(in_features, width), nn.ReLU(inplace=True)))
            # A single light dropout at the end of the head preserves the Stage 1
            # topology while avoiding the earlier over-regularised small-data regime.
            if index == len(hidden_layers) - 1 and classifier_dropout > 0.0:
                layers.append(nn.Dropout(p=classifier_dropout))
            in_features = width
        output_features = 1 if output_mode == "sigmoid1" else 2
        layers.append(nn.Linear(in_features, output_features))
        return nn.Sequential(*layers)

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

    @property
    def output_features(self) -> int:
        return 1 if self.output_mode == "sigmoid1" else 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
