#!/usr/bin/env python3
"""Re-evaluate a saved checkpoint on the provided test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cnn_cats_dogs.engine import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia um checkpoint da ScratchCNN no conjunto de teste.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "re_evaluation")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()

    result = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_name=args.device,
    )
    print(
        f"Teste: loss={result.loss:.4f}, accuracy={result.accuracy:.4f}, "
        f"precision={result.precision:.4f}, recall={result.recall:.4f}, f1={result.f1:.4f}"
    )


if __name__ == "__main__":
    main()
