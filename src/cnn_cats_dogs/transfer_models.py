"""TorchVision backbones and trainability controls for Stage 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import torch
from torch import nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    convnext_tiny,
    efficientnet_b0,
    resnet18,
)

TransferPhase = Literal["head", "finetune"]


@dataclass(frozen=True)
class TransferModelSpec:
    """Metadata and construction rules for one supported pretrained backbone."""

    identifier: str
    display_name: str
    builder: Callable[..., nn.Module]
    weights: Any
    final_stage_name: str
    parameters_millions: float
    flops_giga: float


TRANSFER_MODEL_SPECS: dict[str, TransferModelSpec] = {
    "resnet18": TransferModelSpec(
        identifier="resnet18",
        display_name="ResNet18",
        builder=resnet18,
        weights=ResNet18_Weights.DEFAULT,
        final_stage_name="layer4",
        parameters_millions=11.69,
        flops_giga=1.81,
    ),
    "efficientnet_b0": TransferModelSpec(
        identifier="efficientnet_b0",
        display_name="EfficientNet-B0",
        builder=efficientnet_b0,
        weights=EfficientNet_B0_Weights.DEFAULT,
        final_stage_name="features[-1]",
        parameters_millions=5.29,
        flops_giga=0.39,
    ),
    "convnext_tiny": TransferModelSpec(
        identifier="convnext_tiny",
        display_name="ConvNeXt-Tiny",
        builder=convnext_tiny,
        weights=ConvNeXt_Tiny_Weights.DEFAULT,
        final_stage_name="features[-1]",
        parameters_millions=28.59,
        flops_giga=4.46,
    ),
}


def transfer_model_ids() -> tuple[str, ...]:
    """Return model identifiers in the intended comparison order."""

    return tuple(TRANSFER_MODEL_SPECS)


def get_transfer_model_spec(identifier: str) -> TransferModelSpec:
    """Resolve a supported model before expensive weights are downloaded."""

    try:
        return TRANSFER_MODEL_SPECS[identifier]
    except KeyError as error:
        choices = ", ".join(transfer_model_ids())
        raise ValueError(f"Modelo de transfer learning desconhecido: {identifier!r}. Opções: {choices}.") from error


def create_transfer_model(identifier: str, *, pretrained: bool) -> nn.Module:
    """Construct one model with a two-logit classifier for cats versus dogs.

    The classifier returns raw logits. CrossEntropyLoss owns the Softmax operation
    during optimisation, avoiding a numerically fragile Softmax-then-log sequence.
    """

    spec = get_transfer_model_spec(identifier)
    model = spec.builder(weights=spec.weights if pretrained else None)
    _replace_final_classifier(model, identifier)
    return model


def _replace_final_classifier(model: nn.Module, identifier: str) -> None:
    if identifier == "resnet18":
        if not isinstance(model.fc, nn.Linear):
            raise TypeError("Estrutura inesperada da ResNet18: fc não é Linear.")
        model.fc = nn.Linear(model.fc.in_features, 2)
        return

    if identifier in {"efficientnet_b0", "convnext_tiny"}:
        classifier = model.classifier
        if not isinstance(classifier, nn.Sequential) or not isinstance(classifier[-1], nn.Linear):
            raise TypeError(f"Estrutura inesperada de {identifier}: último classificador não é Linear.")
        classifier[-1] = nn.Linear(classifier[-1].in_features, 2)
        return

    raise AssertionError(f"Sem regra de classificador para {identifier!r}.")


def _classifier_module(model: nn.Module, identifier: str) -> nn.Module:
    if identifier == "resnet18":
        return model.fc
    if identifier in {"efficientnet_b0", "convnext_tiny"}:
        return model.classifier
    raise AssertionError(f"Sem regra de classificador para {identifier!r}.")


def _last_stage_module(model: nn.Module, identifier: str) -> nn.Module:
    if identifier == "resnet18":
        return model.layer4
    if identifier in {"efficientnet_b0", "convnext_tiny"}:
        return model.features[-1]
    raise AssertionError(f"Sem regra de último estágio para {identifier!r}.")


def configure_trainability(model: nn.Module, identifier: str, phase: TransferPhase) -> dict[str, int | str]:
    """Freeze the backbone or unfreeze only its final visual stage.

    ``head`` trains only the new binary classifier. ``finetune`` trains that
    classifier plus the final convolutional stage. Earlier layers remain frozen
    so a few hundred images do not rewrite all ImageNet features in one afternoon.
    """

    if phase not in {"head", "finetune"}:
        raise ValueError("phase precisa ser 'head' ou 'finetune'.")

    for parameter in model.parameters():
        parameter.requires_grad = False

    head = _classifier_module(model, identifier)
    for parameter in head.parameters():
        parameter.requires_grad = True

    trainable_modules = ["classifier"]
    if phase == "finetune":
        stage = _last_stage_module(model, identifier)
        for parameter in stage.parameters():
            parameter.requires_grad = True
        trainable_modules.insert(0, get_transfer_model_spec(identifier).final_stage_name)

    total_parameters = count_parameters(model, trainable_only=False)
    trainable_parameters = count_parameters(model, trainable_only=True)
    return {
        "phase": phase,
        "trainable_modules": ", ".join(trainable_modules),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
    }


def set_frozen_batch_norm_eval(model: nn.Module) -> None:
    """Prevent frozen BatchNorm layers from drifting through small batches.

    Calling ``model.train()`` would otherwise update running mean/variance even
    when the BatchNorm affine parameters are frozen. Only fully frozen BatchNorm
    modules are switched back to evaluation mode; unfrozen final-stage norms may
    still adapt during fine-tuning.
    """

    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            local_parameters = tuple(module.parameters(recurse=False))
            if local_parameters and not any(parameter.requires_grad for parameter in local_parameters):
                module.eval()


def count_parameters(model: nn.Module, *, trainable_only: bool) -> int:
    """Count all or only gradient-enabled parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad or not trainable_only
    )


def transfer_model_metadata(identifier: str) -> dict[str, Any]:
    """Return serialisable metadata used in experiment artifacts and reports."""

    spec = get_transfer_model_spec(identifier)
    weights_transform = spec.weights.transforms()
    crop_size = getattr(weights_transform, "crop_size", [224])
    resize_size = getattr(weights_transform, "resize_size", [256])
    return {
        "identifier": spec.identifier,
        "display_name": spec.display_name,
        "weights": f"{type(spec.weights).__name__}.{spec.weights.name}",
        "weights_url": spec.weights.url,
        "final_stage": spec.final_stage_name,
        "parameters_millions_reference": spec.parameters_millions,
        "flops_giga_reference": spec.flops_giga,
        "evaluation_crop_size": list(crop_size) if isinstance(crop_size, (list, tuple)) else crop_size,
        "evaluation_resize_size": list(resize_size) if isinstance(resize_size, (list, tuple)) else resize_size,
        "normalization_mean": list(getattr(weights_transform, "mean", [])),
        "normalization_std": list(getattr(weights_transform, "std", [])),
        "interpolation": str(getattr(weights_transform, "interpolation", "unknown")),
    }
