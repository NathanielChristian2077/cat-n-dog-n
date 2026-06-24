"""CLI for a final held-out test evaluation of a transfer-learning checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from .transfer_engine import evaluate_transfer_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Avalia uma única vez no teste um checkpoint de transfer learning selecionado por validação."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = evaluate_transfer_checkpoint(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_name=args.device,
    )
    print(
        f"Teste final: loss={result.loss:.4f}, accuracy={result.accuracy:.4f}, "
        f"precision={result.precision:.4f}, recall={result.recall:.4f}, f1={result.f1:.4f}"
    )


if __name__ == "__main__":
    main()
