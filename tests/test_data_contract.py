from pathlib import Path

import pytest
from PIL import Image

from cnn_cats_dogs.config import TrainingConfig
from cnn_cats_dogs.data import build_data_bundle


def _make_rgb_image(path: Path) -> None:
    image = Image.new("RGB", (240, 240), color=(120, 80, 50))
    image.save(path)


def _make_split_first_dataset(root: Path, split_names: dict[str, str], class_names: tuple[str, str]) -> None:
    for canonical in ("train", "val", "test"):
        for class_name in class_names:
            class_dir = root / split_names[canonical] / class_name
            class_dir.mkdir(parents=True)
            _make_rgb_image(class_dir / f"{canonical}_{class_name}.jpg")


def test_binary_folder_contract_remaps_positive_class(tmp_path: Path) -> None:
    _make_split_first_dataset(
        tmp_path,
        {"train": "train", "val": "val", "test": "test"},
        ("cats", "dogs"),
    )
    config = TrainingConfig(
        data_dir=tmp_path,
        output_dir=tmp_path / "out",
        batch_size=1,
        num_workers=0,
        positive_class="dogs",
    )
    bundle = build_data_bundle(config)
    image, label = next(iter(bundle.train_loader))
    assert image.shape == (1, 3, 224, 224)
    assert label.item() in {0, 1}
    assert bundle.class_names == ("cats", "dogs")


def test_nested_portuguese_layout_is_resolved_without_copying(tmp_path: Path) -> None:
    provider_root = tmp_path / "dataset_professor"
    _make_split_first_dataset(
        provider_root,
        {"train": "treino", "val": "validacao", "test": "teste"},
        ("Gatos", "Cachorros"),
    )
    config = TrainingConfig(
        data_dir=tmp_path,
        output_dir=tmp_path / "out",
        batch_size=1,
        num_workers=0,
        positive_class="dogs",
    )
    bundle = build_data_bundle(config)
    assert bundle.dataset_root == provider_root
    assert bundle.sizes == {"train": 2, "val": 2, "test": 2}
    assert bundle.positive_class == "Cachorros"
    assert bundle.negative_class == "Gatos"


def test_layout_requires_professor_splits(tmp_path: Path) -> None:
    config = TrainingConfig(data_dir=tmp_path, output_dir=tmp_path / "out")
    with pytest.raises(FileNotFoundError):
        build_data_bundle(config)
