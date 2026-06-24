"""Dataset loaders and augmentation for TorchVision transfer-learning models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .data import (
    BinaryImageFolder,
    _binary_counts,
    _resolve_split_dirs,
    _validate_binary_semantics,
)
from .transfer_config import TransferTrainingConfig
from .transfer_models import get_transfer_model_spec
from .utils import seed_worker


@dataclass(frozen=True)
class TransferTransforms:
    """Train/eval transforms plus serialisable provenance from the weight preset."""

    train: transforms.Compose
    evaluation: Any
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TransferDataBundle:
    """Loaders for augmentation, clean-train diagnostics, validation, and testing."""

    train_loader: DataLoader
    train_eval_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    negative_class: str
    positive_class: str
    class_names: tuple[str, str]
    sizes: dict[str, int]
    class_counts: dict[str, dict[str, int]]
    dataset_root: Path
    split_dirs: dict[str, Path]
    transform_metadata: dict[str, Any]


def _size_scalar(value: int | list[int] | tuple[int, ...]) -> int:
    if isinstance(value, int):
        return value
    if len(value) != 1:
        raise ValueError(f"Transform de peso inesperado: tamanho não escalar {value!r}.")
    return int(value[0])


def build_transfer_transforms(architecture: str, image_size: int = 224) -> TransferTransforms:
    """Build train augmentation around the exact official evaluation transform.

    Validation and test delegate to ``weights.transforms()``. Training applies
    four mild augmentation groups before the same tensor normalization expected
    by ImageNet weights: crop/scale, horizontal reflection, photometric jitter,
    and random erasing.
    """

    spec = get_transfer_model_spec(architecture)
    evaluation = spec.weights.transforms()
    crop_size = _size_scalar(getattr(evaluation, "crop_size", image_size))
    if crop_size != image_size:
        raise ValueError(
            f"O peso {architecture} espera crop {crop_size}, mas a Etapa 3 exige {image_size}x{image_size}."
        )

    mean = tuple(float(value) for value in getattr(evaluation, "mean"))
    std = tuple(float(value) for value in getattr(evaluation, "std"))
    interpolation = getattr(evaluation, "interpolation")

    train = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.85, 1.00),
                ratio=(0.90, 1.10),
                interpolation=interpolation,
            ),
            transforms.RandomHorizontalFlip(p=0.50),
            transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.08, hue=0.01),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.05, scale=(0.02, 0.06), ratio=(0.50, 2.00), value="random"),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    metadata = {
        "weights": f"{type(spec.weights).__name__}.{spec.weights.name}",
        "evaluation_transform": repr(evaluation),
        "evaluation_crop_size": crop_size,
        "evaluation_resize_size": _size_scalar(getattr(evaluation, "resize_size", image_size)),
        "normalization_mean": list(mean),
        "normalization_std": list(std),
        "interpolation": str(interpolation),
        "train_augmentations": [
            "RandomResizedCrop(scale=0.85..1.00)",
            "RandomHorizontalFlip(p=0.50)",
            "ColorJitter",
            "RandomErasing(p=0.05)",
        ],
    }
    return TransferTransforms(train=train, evaluation=evaluation, metadata=metadata)


def _build_loader(
    dataset: BinaryImageFolder,
    config: TransferTrainingConfig,
    *,
    shuffle: bool,
    generator: torch.Generator | None,
) -> DataLoader:
    pin_memory = str(config.device).startswith("cuda") or (
        config.device == "auto" and torch.cuda.is_available()
    )
    kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": config.batch_size,
        "shuffle": shuffle,
        "num_workers": config.num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker if config.num_workers > 0 else None,
        "generator": generator,
    }
    if config.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = config.prefetch_factor
    return DataLoader(**kwargs)


def build_transfer_data_bundle(config: TransferTrainingConfig) -> TransferDataBundle:
    """Load the professor-provided fixed splits for a selected weight preset."""

    dataset_root, split_dirs = _resolve_split_dirs(config.data_dir)
    prepared_transforms = build_transfer_transforms(config.architecture, config.image_size)

    train_dataset = BinaryImageFolder(
        split_dirs["train"],
        transform=prepared_transforms.train,
        positive_class=config.positive_class,
    )
    train_eval_dataset = BinaryImageFolder(
        split_dirs["train"],
        transform=prepared_transforms.evaluation,
        positive_class=train_dataset.positive_class,
    )
    val_dataset = BinaryImageFolder(
        split_dirs["val"],
        transform=prepared_transforms.evaluation,
        positive_class=train_dataset.positive_class,
    )
    test_dataset = BinaryImageFolder(
        split_dirs["test"],
        transform=prepared_transforms.evaluation,
        positive_class=train_dataset.positive_class,
    )

    for split_name, dataset in (
        ("train_eval", train_eval_dataset),
        ("val", val_dataset),
        ("test", test_dataset),
    ):
        _validate_binary_semantics(train_dataset, dataset, split_name)

    class_counts = {
        "train": _binary_counts(train_dataset, "train"),
        "val": _binary_counts(val_dataset, "val"),
        "test": _binary_counts(test_dataset, "test"),
    }
    generator = torch.Generator().manual_seed(config.seed)
    return TransferDataBundle(
        train_loader=_build_loader(train_dataset, config, shuffle=True, generator=generator),
        train_eval_loader=_build_loader(train_eval_dataset, config, shuffle=False, generator=None),
        val_loader=_build_loader(val_dataset, config, shuffle=False, generator=None),
        test_loader=_build_loader(test_dataset, config, shuffle=False, generator=None),
        negative_class=train_dataset.negative_class,
        positive_class=train_dataset.positive_class,
        class_names=(train_dataset.negative_class, train_dataset.positive_class),
        sizes={"train": len(train_dataset), "val": len(val_dataset), "test": len(test_dataset)},
        class_counts=class_counts,
        dataset_root=dataset_root,
        split_dirs=split_dirs,
        transform_metadata=prepared_transforms.metadata,
    )
