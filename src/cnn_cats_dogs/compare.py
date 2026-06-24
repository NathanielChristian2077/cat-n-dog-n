"""Run the ranked Stage 1 head variants under one controlled Stage 2 protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import PHASE1_RANKED_PRESETS, TrainingConfig, get_classifier_preset
from .engine import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compara heads CNN derivados das arquiteturas vencedoras da Etapa 1."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--positive-class", default="dogs")
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=PHASE1_RANKED_PRESETS,
        default=list(PHASE1_RANKED_PRESETS),
        help="Por padrão executa os seis presets ranqueados da Etapa 1.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 73, 101],
        help="Seeds independentes; três são o padrão para não confundir sorte com arquitetura.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--classifier-dropout", type=float, default=0.10)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--disable-amp", action="store_true")
    return parser


def _summary_row(*, architecture: str, seed: int, summary_path: Path) -> dict[str, object]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    test = payload["test_metrics"]
    return {
        "architecture": architecture,
        "stage1_rank": payload["architecture"]["stage1_rank"],
        "head_hidden_layers": "x".join(map(str, payload["architecture"]["head_hidden_layers"])),
        "output_mode": payload["architecture"]["output_mode"],
        "seed": seed,
        "best_epoch": payload["best_epoch"],
        "epochs_completed": payload["epochs_completed"],
        "best_validation_loss": payload["best_validation_loss"],
        "test_loss": test["loss"],
        "test_accuracy": test["accuracy"],
        "test_precision": test["precision"],
        "test_recall": test["recall"],
        "test_f1": test["f1"],
        "training_time_seconds": payload["training_time_seconds"],
        "run_summary": str(summary_path),
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    total_runs = len(args.architectures) * len(args.seeds)
    current_run = 0

    for architecture in args.architectures:
        preset = get_classifier_preset(architecture)
        for seed in args.seeds:
            current_run += 1
            run_dir = args.output_dir / architecture / f"seed_{seed}"
            print(
                f"\n[{current_run}/{total_runs}] Etapa 1 rank {preset.stage1_rank}: "
                f"{preset.hidden_layers}, {preset.output_mode}, seed={seed}"
            )
            config = TrainingConfig(
                data_dir=args.data_dir,
                output_dir=run_dir,
                positive_class=args.positive_class,
                architecture=architecture,
                classifier_dropout=args.classifier_dropout,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                num_workers=args.num_workers,
                seed=seed,
                device=args.device,
                use_amp=not args.disable_amp,
            )
            artifacts = run_training(config)
            rows.append(
                _summary_row(
                    architecture=architecture,
                    seed=seed,
                    summary_path=artifacts.summary_path,
                )
            )
            pd.DataFrame(rows).to_csv(args.output_dir / "comparison_runs.csv", index=False)

    runs = pd.DataFrame(rows)
    score_columns = [
        "best_validation_loss",
        "test_loss",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "training_time_seconds",
    ]
    summary = (
        runs.groupby(["architecture", "stage1_rank", "head_hidden_layers", "output_mode"], as_index=False)[score_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    summary = summary.sort_values(
        by=["test_f1_mean", "test_accuracy_mean", "best_validation_loss_mean"],
        ascending=[False, False, True],
    )
    summary.to_csv(args.output_dir / "comparison_summary.csv", index=False)
    print("\nComparação concluída.")
    print(runs[["architecture", "seed", "test_accuracy", "test_f1", "best_validation_loss"]].to_string(index=False))
    print(f"\nDetalhes por execução: {args.output_dir / 'comparison_runs.csv'}")
    print(f"Resumo agregado: {args.output_dir / 'comparison_summary.csv'}")


if __name__ == "__main__":
    main()
