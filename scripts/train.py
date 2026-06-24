#!/usr/bin/env python3
"""Train the scratch CNN for the IFSC cats-versus-dogs dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cnn_cats_dogs.config import TrainingConfig
from cnn_cats_dogs.engine import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Treina uma CNN própria em PyTorch, sem arquitetura pré-treinada."
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Pasta com train/, val/ e test/.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "cnn_scratch")
    parser.add_argument("--positive-class", default="dogs", help="Nome da pasta considerada classe 1.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--disable-amp", action="store_true", help="Desativa AMP mesmo em CUDA.")
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        positive_class=args.positive_class,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        use_amp=not args.disable_amp,
        early_stopping_patience=args.early_stopping_patience,
    )
    artifacts = run_training(config)
    result = artifacts.test_result
    print("\nTreinamento concluído.")
    print(f"Checkpoint selecionado: {artifacts.best_checkpoint}")
    print(f"Resultados de teste: loss={result.loss:.4f}, accuracy={result.accuracy:.4f}, "
          f"precision={result.precision:.4f}, recall={result.recall:.4f}, f1={result.f1:.4f}")
    print(f"Resumo completo: {artifacts.summary_path}")


if __name__ == "__main__":
    main()
