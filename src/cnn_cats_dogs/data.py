"""Dataset validation, augmentation, and DataLoader construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import TrainingConfig
from .utils import seed_worker

# The scratch CNN is trained from random initialization, so these are neutral
# normalization constants rather than statistics borrowed from a pretrained model.
RGB_MEAN = (0.5, 0.5, 0.5)
RGB_STD = (0.5, 0.5, 0.5)
SPLITS = ("train", "val", "test")


class BinaryImageFolder(datasets.ImageFolder):
    """ImageFolder that remaps any two class folders into binary labels {0, 1}.

    The positive class is explicit instead of relying on alphabetical folder order.
    That avoids the charming academic classic where labels silently flip after a
    teammate renames ``dogs`` to ``cachorros``.
    """

    def __init__(
        self,
        root: str | Path,
        transform: Callable | None,
        positive_class: str,
    ) -> None:
        super().__init__(root=str(root), transform=transform)
        if len(self.classes) != 2:
            raise ValueError(
                f"{root} precisa conter exatamente duas pastas de classe; encontrado: {self.classes}."
            )
        resolved = _resolve_class_name(self.classes, positive_class)
        self.positive_class = resolved
        self.positive_native_index = self.class_to_idx[resolved]
        self.negative_class = next(name for name in self.classes if name != resolved)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, native_target = super().__getitem__(index)
        target = int(native_target == self.positive_native_index)
        return image, target


def _resolve_class_name(classes: list[str], requested: str) -> str:
    exact = {name.casefold(): name for name in classes}
    key = requested.casefold().strip()
    if key in exact:
        return exact[key]

    aliases = {
        "dog": {"dog", "dogs", "cachorro", "cachorros"},
        "cat": {"cat", "cats", "gato", "gatos"},
    }
    for alias_group in aliases.values():
        if key in alias_group:
            matches = [name for name in classes if name.casefold() in alias_group]
            if len(matches) == 1:
                return matches[0]

    raise ValueError(
        f"positive_class={requested!r} não corresponde às classes encontradas: {classes}. "
        "Passe exatamente o nome da pasta positiva com --positive-class."
    )


def build_train_transform(image_size: int) -> transforms.Compose:
    """Return the required RGB 224x224 preprocessing plus four augmentation groups.

    Augmentations used here:
      1. RandomResizedCrop: crop/scale spatial;
      2. RandomHorizontalFlip: reflection geometry;
      3. ColorJitter: photometric color/intensity;
      4. RandomErasing: random occlusion/regularization.

    The assignment asks for at least three techniques from different groups; four
    are configured so the report can identify them without playing word games.
    """

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.80, 1.00), ratio=(0.90, 1.10)),
            transforms.RandomHorizontalFlip(p=0.50),
            transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15, hue=0.02),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.20, scale=(0.02, 0.12), ratio=(0.30, 3.30), value="random"),
            transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
        ]
    )


def build_eval_transform(image_size: int) -> transforms.Compose:
    """Deterministic preprocessing for validation and test."""

    resize_size = int(round(image_size * 1.14))
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
        ]
    )


@dataclass(frozen=True)
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    negative_class: str
    positive_class: str
    class_names: tuple[str, str]
    sizes: dict[str, int]


def _validate_split_layout(data_dir: Path) -> None:
    missing = [split for split in SPLITS if not (data_dir / split).is_dir()]
    if missing:
        expected = "\n".join(f"  {data_dir / split}/<classe>/imagem.jpg" for split in SPLITS)
        raise FileNotFoundError(
            "Divisões ausentes: "
            f"{missing}. A Etapa 2 deve usar a divisão fornecida pelo professor, sem criar uma "
            f"divisão nova. Estrutura esperada:\n{expected}"
        )


def _build_loader(
    dataset: BinaryImageFolder,
    config: TrainingConfig,
    *,
    shuffle: bool,
    generator: torch.Generator | None,
) -> DataLoader:
    kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": config.batch_size,
        "shuffle": shuffle,
        "num_workers": config.num_workers,
        "pin_memory": config.device == "cuda" or (config.device == "auto" and torch.cuda.is_available()),
        "worker_init_fn": seed_worker if config.num_workers > 0 else None,
        "generator": generator,
    }
    if config.num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(**kwargs)


def build_data_bundle(config: TrainingConfig) -> DataBundle:
    """Load professor-provided train/val/test folders with strict consistency checks."""

    _validate_split_layout(config.data_dir)
    train_dataset = BinaryImageFolder(
        config.data_dir / "train",
        transform=build_train_transform(config.image_size),
        positive_class=config.positive_class,
    )
    val_dataset = BinaryImageFolder(
        config.data_dir / "val",
        transform=build_eval_transform(config.image_size),
        positive_class=train_dataset.positive_class,
    )
    test_dataset = BinaryImageFolder(
        config.data_dir / "test",
        transform=build_eval_transform(config.image_size),
        positive_class=train_dataset.positive_class,
    )

    expected_classes = set(train_dataset.classes)
    for split_name, dataset in (("val", val_dataset), ("test", test_dataset)):
        if set(dataset.classes) != expected_classes:
            raise ValueError(
                f"Classes de {split_name}={dataset.classes} diferem do treino={train_dataset.classes}. "
                "A mesma taxonomia precisa existir nas três divisões."
            )
        if dataset.positive_class != train_dataset.positive_class:
            raise ValueError("A classe positiva mudou entre os splits, o que não deveria ser possível.")

    generator = torch.Generator().manual_seed(config.seed)
    train_loader = _build_loader(train_dataset, config, shuffle=True, generator=generator)
    val_loader = _build_loader(val_dataset, config, shuffle=False, generator=None)
    test_loader = _build_loader(test_dataset, config, shuffle=False, generator=None)

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        negative_class=train_dataset.negative_class,
        positive_class=train_dataset.positive_class,
        class_names=(train_dataset.negative_class, train_dataset.positive_class),
        sizes={"train": len(train_dataset), "val": len(val_dataset), "test": len(test_dataset)},
    )
