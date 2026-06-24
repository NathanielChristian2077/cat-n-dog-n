"""Audit a fixed image split for duplicate and near-duplicate leakage."""

from __future__ import annotations

import csv
import hashlib
import itertools
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps
from torchvision.datasets.folder import IMG_EXTENSIONS

from .data import _resolve_split_dirs
from .utils import write_json


@dataclass(frozen=True)
class ImageRecord:
    """One image with enough context to audit cross-split overlap."""

    split: str
    class_name: str
    relative_path: str
    sha256: str
    dhash: int


def _iter_image_paths(split: str, split_dir: Path, dataset_root: Path) -> Iterable[tuple[str, str, Path]]:
    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        for path in sorted(candidate for candidate in class_dir.rglob("*") if candidate.is_file()):
            if path.suffix.casefold() not in IMG_EXTENSIONS:
                continue
            yield split, class_dir.name, path.relative_to(dataset_root)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dhash(path: Path, *, hash_size: int = 8) -> int:
    """Return a compact perceptual hash insensitive to modest resize/compression.

    A dHash collision is only a review candidate, not proof of leakage. Exact
    SHA-256 matches are proof that the underlying bytes were repeated.
    """

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L")
        resized = image.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(resized.getdata())
    result = 0
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for col in range(hash_size):
            result = (result << 1) | int(pixels[offset + col] > pixels[offset + col + 1])
    return result


def _hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _read_records(dataset_root: Path, split_dirs: dict[str, Path]) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for split, split_dir in split_dirs.items():
        for split_name, class_name, relative_path in _iter_image_paths(split, split_dir, dataset_root):
            absolute_path = dataset_root / relative_path
            records.append(
                ImageRecord(
                    split=split_name,
                    class_name=class_name,
                    relative_path=str(relative_path),
                    sha256=_sha256(absolute_path),
                    dhash=_dhash(absolute_path),
                )
            )
    return records


def _write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _exact_duplicate_rows(records: list[ImageRecord]) -> list[dict[str, object]]:
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.sha256].append(record)

    rows: list[dict[str, object]] = []
    for digest, matches in grouped.items():
        for left, right in itertools.combinations(matches, 2):
            if left.split == right.split:
                continue
            rows.append(
                {
                    "sha256": digest,
                    "left_split": left.split,
                    "left_class": left.class_name,
                    "left_path": left.relative_path,
                    "right_split": right.split,
                    "right_class": right.class_name,
                    "right_path": right.relative_path,
                    "same_class_name": left.class_name == right.class_name,
                }
            )
    return rows


def _near_duplicate_rows(records: list[ImageRecord], *, threshold: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for left, right in itertools.combinations(records, 2):
        if left.split == right.split or left.sha256 == right.sha256:
            continue
        distance = _hamming_distance(left.dhash, right.dhash)
        if distance <= threshold:
            rows.append(
                {
                    "hamming_distance": distance,
                    "left_split": left.split,
                    "left_class": left.class_name,
                    "left_path": left.relative_path,
                    "right_split": right.split,
                    "right_class": right.class_name,
                    "right_path": right.relative_path,
                    "same_class_name": left.class_name == right.class_name,
                }
            )
    return sorted(rows, key=lambda row: int(row["hamming_distance"]))


def audit_dataset(
    *,
    data_dir: Path,
    output_dir: Path,
    dhash_threshold: int = 4,
) -> dict[str, object]:
    """Audit exact and perceptual overlap across the professor-provided splits."""

    if not 0 <= dhash_threshold <= 64:
        raise ValueError("dhash_threshold precisa estar entre 0 e 64.")

    dataset_root, split_dirs = _resolve_split_dirs(data_dir)
    records = _read_records(dataset_root, split_dirs)
    expected_splits = {"train", "val", "test"}
    found_splits = {record.split for record in records}
    if found_splits != expected_splits:
        raise ValueError(f"Auditoria esperava os splits {expected_splits}, encontrou {found_splits}.")

    exact_rows = _exact_duplicate_rows(records)
    near_rows = _near_duplicate_rows(records, threshold=dhash_threshold)
    manifest_rows = [
        {
            **asdict(record),
            "dhash_hex": f"{record.dhash:016x}",
        }
        for record in records
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(
        output_dir / "image_manifest.csv",
        manifest_rows,
        ["split", "class_name", "relative_path", "sha256", "dhash", "dhash_hex"],
    )
    _write_rows(
        output_dir / "exact_cross_split_duplicates.csv",
        exact_rows,
        [
            "sha256",
            "left_split",
            "left_class",
            "left_path",
            "right_split",
            "right_class",
            "right_path",
            "same_class_name",
        ],
    )
    _write_rows(
        output_dir / "near_cross_split_candidates.csv",
        near_rows,
        [
            "hamming_distance",
            "left_split",
            "left_class",
            "left_path",
            "right_split",
            "right_class",
            "right_path",
            "same_class_name",
        ],
    )

    counts_by_split: dict[str, int] = defaultdict(int)
    for record in records:
        counts_by_split[record.split] += 1
    summary: dict[str, object] = {
        "dataset_root": str(dataset_root),
        "split_dirs": {name: str(path) for name, path in split_dirs.items()},
        "images_by_split": dict(sorted(counts_by_split.items())),
        "images_total": len(records),
        "exact_cross_split_pairs": len(exact_rows),
        "exact_cross_split_label_conflicts": sum(not bool(row["same_class_name"]) for row in exact_rows),
        "near_duplicate_dhash_threshold": dhash_threshold,
        "near_cross_split_candidate_pairs": len(near_rows),
        "near_cross_split_label_conflicts": sum(not bool(row["same_class_name"]) for row in near_rows),
        "interpretation": {
            "exact_duplicates": "Qualquer linha indica repetição byte-a-byte entre splits e deve ser investigada.",
            "near_candidates": "Candidatos por dHash requerem inspeção visual; não são prova isolada de vazamento.",
        },
        "artifacts": {
            "manifest": str(output_dir / "image_manifest.csv"),
            "exact_duplicates": str(output_dir / "exact_cross_split_duplicates.csv"),
            "near_candidates": str(output_dir / "near_cross_split_candidates.csv"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary
