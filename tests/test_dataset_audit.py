from pathlib import Path

from PIL import Image

from cnn_cats_dogs.dataset_audit import audit_dataset


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path)


def test_dataset_audit_reports_exact_cross_split_duplicate(tmp_path: Path) -> None:
    data = tmp_path / "data"
    train_image = data / "train" / "cats" / "same.png"
    _write_image(train_image, (180, 80, 40))
    _write_image(data / "train" / "dogs" / "dog.png", (20, 60, 140))

    val_copy = data / "val" / "cats" / "same.png"
    val_copy.parent.mkdir(parents=True, exist_ok=True)
    val_copy.write_bytes(train_image.read_bytes())
    _write_image(data / "val" / "dogs" / "dog.png", (40, 100, 180))

    _write_image(data / "test" / "cats" / "cat.png", (100, 130, 60))
    _write_image(data / "test" / "dogs" / "dog.png", (80, 30, 160))

    output = tmp_path / "audit"
    summary = audit_dataset(data_dir=data, output_dir=output)

    assert summary["exact_cross_split_pairs"] == 1
    assert (output / "exact_cross_split_duplicates.csv").is_file()
    assert (output / "near_cross_split_candidates.csv").is_file()
    assert (output / "summary.json").is_file()
