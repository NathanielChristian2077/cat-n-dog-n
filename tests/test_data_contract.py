from pathlib import Path

import pytest
from PIL import Image

from cnn_cats_dogs.config import TrainingConfig
from cnn_cats_dogs.data import build_data_bundle


def _make_rgb_image(path: Path) -> None:
    image = Image.new("RGB", (240, 240), color=(120, 80, 50))
    image.save(path)


def test_binary_folder_contract_remaps_positive_class(tmp_path: Path) -> None:
    for split in ("train", "val", "test"):
        for class_name in ("cats", "dogs"):
            class_dir = tmp_path / split / class_name
            class_dir.mkdir(parents=True)
            _make_rgb_image(class_dir / "sample.jpg")

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


def test_layout_requires_professor_splits(tmp_path: Path) -> None:
    config = TrainingConfig(data_dir=tmp_path, output_dir=tmp_path / "out")
    with pytest.raises(FileNotFoundError):
        build_data_bundle(config)
