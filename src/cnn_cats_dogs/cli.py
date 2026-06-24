"""Command-line entry point for comparable Stage 2 CNN experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import TrainingConfig, architecture_ids
from .engine import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Treina uma CNN própria com head derivado das melhores MLPs da Etapa 1."
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Pasta com train/, val/ e test/.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Diretório exclusivo deste experimento.")
    parser.add_argument("--positive-class", default="dogs", help="Nome da pasta considerada classe 1.")
    parser.add_argument(
        "--architecture",
        choices=architecture_ids(),
        default="phase1_rank1_32x64x512_softmax2",
        help="Topologia densa comparável à Etapa 1.",
    )
    parser.add_argument("--classifier-dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--disable-amp", action="store_true", help="Desativa AMP mesmo em CUDA.")
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        positive_class=args.positive_class,
        architecture=args.architecture,
        classifier_dropout=args.classifier_dropout,
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
    print(
        f"Resultados de teste: loss={result.loss:.4f}, accuracy={result.accuracy:.4f}, "
        f"precision={result.precision:.4f}, recall={result.recall:.4f}, f1={result.f1:.4f}"
    )
    print(f"Resumo completo: {artifacts.summary_path}")


if __name__ == "__main__":
    main()
