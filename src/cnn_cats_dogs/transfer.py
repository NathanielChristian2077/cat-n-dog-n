"""CLI entry point for a single Stage 3 transfer-learning run."""

from __future__ import annotations

import argparse
from pathlib import Path

from .transfer_config import TransferTrainingConfig
from .transfer_engine import run_transfer_training
from .transfer_models import transfer_model_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Treina uma CNN pré-treinada do TorchVision em duas fases de fine-tuning."
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Pasta com train/, val/ e test/.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Diretório exclusivo do experimento.")
    parser.add_argument("--architecture", choices=transfer_model_ids(), required=True)
    parser.add_argument("--positive-class", default="dogs", help="Nome da pasta considerada classe 1.")
    parser.add_argument("--head-epochs", type=int, default=12)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--finetune-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--disable-amp", action="store_true", help="Desativa AMP mesmo em CUDA.")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Avalia o melhor checkpoint no teste. Use somente após fechar decisões por validação.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = TransferTrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        architecture=args.architecture,
        positive_class=args.positive_class,
        head_epochs=args.head_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        head_learning_rate=args.head_learning_rate,
        finetune_learning_rate=args.finetune_learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        use_amp=not args.disable_amp,
        evaluate_test=args.evaluate_test,
    )
    artifacts = run_transfer_training(config)
    print("\nTreinamento concluído.")
    print(f"Checkpoint selecionado por validação: {artifacts.best_checkpoint}")
    if artifacts.test_result is None:
        print("Teste não executado nesta rodada. Use transfer_evaluate.py após fechar a seleção.")
    else:
        result = artifacts.test_result
        print(
            f"Resultados de teste: loss={result.loss:.4f}, accuracy={result.accuracy:.4f}, "
            f"precision={result.precision:.4f}, recall={result.recall:.4f}, f1={result.f1:.4f}"
        )
    print(f"Resumo completo: {artifacts.summary_path}")


if __name__ == "__main__":
    main()
