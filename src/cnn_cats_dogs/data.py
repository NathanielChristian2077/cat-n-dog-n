"""Dataset discovery, augmentation, and DataLoader construction."""

from __future__ import annotations

import re
import unicodedata
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
SPLIT_ALIASES = {
    "train": {"train", "training", "treino", "treinamento", "training_set", "treino_set"},
    "val": {"val", "valid", "validation", "validacao", "validacao_set", "validation_set", "dev"},
    "test": {"test", "testing", "teste", "test_set", "testing_set", "teste_set"},
}
CLASS_ALIASES = {
    "dog": {"dog", "dogs", "cachorro", "cachorros"},
    "cat": {"cat", "cats", "gato", "gatos"},
}


def _normalise_name(value: str) -> str:
    """Normalise harmless naming differences without changing the data itself."""

    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "_", without_accents).strip("_")


def _canonical_split(name: str) -> str | None:
    normalized = _normalise_name(name)
    for canonical, aliases in SPLIT_ALIASES.items():
        if normalized in aliases:
            return canonical
    return None


def _class_family(name: str) -> str | None:
    normalized = _normalise_name(name)
    for family, aliases in CLASS_ALIASES.items():
        if normalized in aliases:
            return family
    return None


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
    exact = {_normalise_name(name): name for name in classes}
    key = _normalise_name(requested)
    if key in exact:
        return exact[key]

    requested_family = _class_family(requested)
    if requested_family is not None:
        matches = [name for name in classes if _class_family(name) == requested_family]
        if len(matches) == 1:
            return matches[0]

    raise ValueError(
        f"positive_class={requested!r} não corresponde às classes encontradas: {classes}. "
        "Passe o nome da pasta positiva ou um alias reconhecido, como dogs/cachorros."
    )


def _resolve_split_dirs(data_dir: Path) -> tuple[Path, dict[str, Path]]:
    """Locate a professor-provided split-first layout without moving any files.

    Accepted examples include ``train/val/test``, ``treino/validacao/teste`` and
    a single enclosing directory such as ``dataset/cats_dogs/{train,val,test}``.
    The assignment's division remains untouched; only names are normalised.
    """

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset não encontrado: {data_dir}")

    candidate_roots = [data_dir]
    candidate_roots.extend(sorted(path for path in data_dir.iterdir() if path.is_dir()))

    partial_layouts: list[tuple[Path, dict[str, Path]]] = []
    for root in candidate_roots:
        resolved: dict[str, Path] = {}
        for child in sorted(path for path in root.iterdir() if path.is_dir()):
            canonical = _canonical_split(child.name)
            if canonical is None:
                continue
            if canonical in resolved:
                raise ValueError(
                    f"Foram encontradas duas pastas para o split {canonical!r} em {root}: "
                    f"{resolved[canonical].name!r} e {child.name!r}. Remova a ambiguidade."
                )
            resolved[canonical] = child
        if set(resolved) == set(SPLITS):
            return root, resolved
        if resolved:
            partial_layouts.append((root, resolved))

    found = "; ".join(
        f"{root}: {', '.join(sorted(layout))}" for root, layout in partial_layouts
    ) or "nenhum split reconhecido"
    aliases = "train/val/test, training/validation/testing ou treino/validacao/teste"
    raise FileNotFoundError(
        f"Não foi possível localizar os três splits em {data_dir}. Encontrado: {found}. "
        f"Use uma estrutura split-first com nomes equivalentes a {aliases}."
    )


def _validate_binary_semantics(train_dataset: BinaryImageFolder, other: BinaryImageFolder, split: str) -> None:
    """Accept translated class names, but reject a genuinely different taxonomy."""

    if set(train_dataset.classes) == set(other.classes):
        return
    train_families = {_class_family(name) for name in train_dataset.classes}
    other_families = {_class_family(name) for name in other.classes}
    if train_families == {"cat", "dog"} and other_families == {"cat", "dog"}:
        return
    raise ValueError(
        f"Classes de {split}={other.classes} diferem da taxonomia de treino={train_dataset.classes}. "
        "Os três splits precisam representar as mesmas duas classes."
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
    dataset_root: Path
    split_dirs: dict[str, Path]


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
        kwargs["prefetch_factor"] = config.prefetch_factor
    return DataLoader(**kwargs)


def build_data_bundle(config: TrainingConfig) -> DataBundle:
    """Load the professor-provided splits with strict, name-tolerant validation."""

    dataset_root, split_dirs = _resolve_split_dirs(config.data_dir)
    train_dataset = BinaryImageFolder(
        split_dirs["train"],
        transform=build_train_transform(config.image_size),
        positive_class=config.positive_class,
    )
    val_dataset = BinaryImageFolder(
        split_dirs["val"],
        transform=build_eval_transform(config.image_size),
        positive_class=config.positive_class,
    )
    test_dataset = BinaryImageFolder(
        split_dirs["test"],
        transform=build_eval_transform(config.image_size),
        positive_class=config.positive_class,
    )

    for split_name, dataset in (("val", val_dataset), ("test", test_dataset)):
        _validate_binary_semantics(train_dataset, dataset, split_name)
        if _class_family(train_dataset.positive_class) and _class_family(dataset.positive_class):
            if _class_family(train_dataset.positive_class) != _class_family(dataset.positive_class):
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
        dataset_root=dataset_root,
        split_dirs=split_dirs,
    )
