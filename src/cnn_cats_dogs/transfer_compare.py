"""Compare Stage 3 pretrained backbones without exposing the test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .transfer_config import TransferTrainingConfig
from .transfer_engine import run_transfer_training
from .transfer_models import get_transfer_model_spec, transfer_model_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compara ResNet18, EfficientNet-B0 e ConvNeXt-Tiny apenas pela validação."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--positive-class", default="dogs")
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=transfer_model_ids(),
        default=list(transfer_model_ids()),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 73, 101])
    parser.add_argument("--head-epochs", type=int, default=12)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--finetune-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--disable-amp", action="store_true")
    return parser


def _summary_row(*, architecture: str, seed: int, summary_path: Path) -> dict[str, object]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    selected = payload["selected_checkpoint"]
    return {
        "architecture": architecture,
        "display_name": payload["architecture"]["display_name"],
        "seed": seed,
        "selected_phase": selected["phase"],
        "selected_global_epoch": selected["global_epoch"],
        "best_validation_loss": selected["best_validation_loss"],
        "head_best_validation_loss": payload["phases"]["head"]["best_validation_loss"],
        "finetune_best_validation_loss": payload["phases"]["finetune"]["best_validation_loss"],
        "training_time_seconds": payload["training_time_seconds"],
        "summary_path": str(summary_path),
    }


def _aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "best_validation_loss",
        "head_best_validation_loss",
        "finetune_best_validation_loss",
        "training_time_seconds",
    ]
    summary = runs.groupby(["architecture", "display_name"])[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    return summary.sort_values(
        by=["best_validation_loss_mean", "best_validation_loss_std"],
        ascending=[True, True],
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    total_runs = len(args.architectures) * len(args.seeds)
    current_run = 0

    for architecture in args.architectures:
        spec = get_transfer_model_spec(architecture)
        for seed in args.seeds:
            current_run += 1
            run_dir = args.output_dir / architecture / f"seed_{seed}"
            print(
                f"\n[{current_run}/{total_runs}] {spec.display_name}, seed={seed}. "
                "Teste ficará intacto até a escolha final por validação."
            )
            config = TransferTrainingConfig(
                data_dir=args.data_dir,
                output_dir=run_dir,
                architecture=architecture,
                positive_class=args.positive_class,
                head_epochs=args.head_epochs,
                finetune_epochs=args.finetune_epochs,
                batch_size=args.batch_size,
                head_learning_rate=args.head_learning_rate,
                finetune_learning_rate=args.finetune_learning_rate,
                weight_decay=args.weight_decay,
                num_workers=args.num_workers,
                seed=seed,
                device=args.device,
                use_amp=not args.disable_amp,
                evaluate_test=False,
            )
            artifacts = run_transfer_training(config)
            rows.append(_summary_row(architecture=architecture, seed=seed, summary_path=artifacts.summary_path))
            pd.DataFrame(rows).to_csv(args.output_dir / "comparison_runs.csv", index=False)

    runs = pd.DataFrame(rows)
    summary = _aggregate(runs)
    summary.to_csv(args.output_dir / "comparison_summary.csv", index=False)
    print("\nComparação de validação concluída.")
    print(runs[["architecture", "seed", "selected_phase", "best_validation_loss"]].to_string(index=False))
    print(f"\nExecuções individuais: {args.output_dir / 'comparison_runs.csv'}")
    print(f"Resumo agregado: {args.output_dir / 'comparison_summary.csv'}")


if __name__ == "__main__":
    main()
