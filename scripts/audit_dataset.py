#!/usr/bin/env python3
"""Audit cats-versus-dogs train/validation/test integrity before trusting results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cnn_cats_dogs.dataset_audit import audit_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Procura duplicatas exatas e candidatos perceptualmente similares entre train/val/test."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "dataset_audit")
    parser.add_argument(
        "--dhash-threshold",
        type=int,
        default=4,
        help="Distância máxima de dHash para candidato de revisão visual, de 0 a 64.",
    )
    args = parser.parse_args()
    summary = audit_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dhash_threshold=args.dhash_threshold,
    )
    print("Auditoria concluída.")
    print(f"Imagens: {summary['images_by_split']} | total={summary['images_total']}")
    print(f"Duplicatas exatas entre splits: {summary['exact_cross_split_pairs']}")
    print(
        "Candidatos perceptuais entre splits "
        f"(dHash <= {summary['near_duplicate_dhash_threshold']}): "
        f"{summary['near_cross_split_candidate_pairs']}"
    )
    print(f"Resumo: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
