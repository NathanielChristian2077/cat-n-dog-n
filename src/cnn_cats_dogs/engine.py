"""Training, validation, checkpointing, and experiment orchestration."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .data import DataBundle, build_data_bundle
from .metrics import EvaluationResult, make_evaluation_result
from .model import ScratchCNN, count_trainable_parameters
from .utils import (
    atomic_torch_save,
    resolve_device,
    resolve_num_workers,
    runtime_snapshot,
    set_global_seed,
    write_json,
)
from .visualization import plot_confusion_matrix, plot_learning_curves


@dataclass(frozen=True)
class RunArtifacts:
    output_dir: Path
    best_checkpoint: Path
    history_path: Path
    summary_path: Path
    test_result: EvaluationResult


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _move_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    *,
    channels_last: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    non_blocking = device.type == "cuda"
    if channels_last and device.type == "cuda":
        images = images.to(device, non_blocking=non_blocking, memory_format=torch.channels_last)
    else:
        images = images.to(device, non_blocking=non_blocking)
    labels = labels.to(device, non_blocking=non_blocking).float().unsqueeze(1)
    return images, labels


def _run_loader(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp_enabled: bool = False,
    max_grad_norm: float | None = None,
    channels_last: bool = False,
) -> EvaluationResult:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_samples = 0
    all_targets: list[int] = []
    all_probabilities: list[float] = []

    for images, labels in loader:
        images, labels = _move_batch(images, labels, device, channels_last=channels_last)
        if is_training:
            optimizer.zero_grad(set_to_none=True)

        grad_context = torch.enable_grad() if is_training else torch.inference_mode()
        with grad_context:
            with _autocast_context(device, amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

        batch_size = labels.size(0)
        total_samples += batch_size
        total_loss += float(loss.detach().item()) * batch_size
        all_targets.extend(labels.detach().squeeze(1).cpu().to(torch.int64).tolist())
        all_probabilities.extend(torch.sigmoid(logits.detach()).squeeze(1).float().cpu().tolist())

    return make_evaluation_result(
        mean_loss=total_loss / max(total_samples, 1),
        targets=all_targets,
        probabilities=all_probabilities,
    )


def _make_optimizer(model: nn.Module, config: TrainingConfig, device: torch.device) -> tuple[AdamW, str]:
    """Use fused AdamW when the local CUDA build supports it, then fall back safely."""

    base_kwargs = {"lr": config.learning_rate, "weight_decay": config.weight_decay}
    if device.type == "cuda":
        try:
            return AdamW(model.parameters(), fused=True, **base_kwargs), "AdamW(fused=True)"
        except (RuntimeError, TypeError):
            pass
    return AdamW(model.parameters(), **base_kwargs), "AdamW"


def _make_training_criterion(
    bundle: DataBundle,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[nn.Module, dict[str, float | bool]]:
    """Weight the minority positive class without altering the professor's split.

    A split can legitimately be imbalanced, but an unweighted BCE then has a
    cheap local optimum: predict the majority class for every image. ``pos_weight
    = negatives / positives`` makes that constant solution sit near 0.5 instead
    of quietly rewarding the prior. Validation and test remain unweighted.
    """

    counts = bundle.class_counts["train"]
    if not config.balance_positive_class:
        return nn.BCEWithLogitsLoss(), {"enabled": False, "pos_weight": 1.0}

    pos_weight = counts["negative"] / counts["positive"]
    weight_tensor = torch.tensor([pos_weight], device=device, dtype=torch.float32)
    return nn.BCEWithLogitsLoss(pos_weight=weight_tensor), {
        "enabled": True,
        "pos_weight": float(pos_weight),
    }


def _checkpoint_payload(
    *,
    model: ScratchCNN,
    optimizer: AdamW,
    scheduler: ReduceLROnPlateau,
    config: TrainingConfig,
    bundle: DataBundle,
    epoch: int,
    best_val_loss: float,
    val_result: EvaluationResult,
) -> dict[str, Any]:
    return {
        "format_version": 3,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config.as_dict(),
        "class_names": list(bundle.class_names),
        "positive_class": bundle.positive_class,
        "class_counts": bundle.class_counts,
        "dataset_root": str(bundle.dataset_root),
        "split_dirs": {name: str(path) for name, path in bundle.split_dirs.items()},
        "validation_metrics": val_result.compact_dict(),
    }


def _history_row(
    epoch: int,
    train: EvaluationResult,
    val: EvaluationResult,
    learning_rate: float,
    epoch_time_seconds: float,
    max_gpu_memory_mb: float | None,
) -> dict[str, float | int | None]:
    return {
        "epoch": epoch,
        "learning_rate": learning_rate,
        "train_loss": train.loss,
        "train_accuracy": train.accuracy,
        "train_precision": train.precision,
        "train_recall": train.recall,
        "train_f1": train.f1,
        "val_loss": val.loss,
        "val_accuracy": val.accuracy,
        "val_precision": val.precision,
        "val_recall": val.recall,
        "val_f1": val.f1,
        "epoch_time_seconds": epoch_time_seconds,
        "max_gpu_memory_mb": max_gpu_memory_mb,
    }


def _prepare_runtime_config(config: TrainingConfig) -> tuple[TrainingConfig, torch.device]:
    device = resolve_device(config.device)
    resolved_workers = resolve_num_workers(config.num_workers, device)
    prepared = replace(config, device=str(device), num_workers=resolved_workers)
    prepared.validate()
    set_global_seed(
        prepared.seed,
        deterministic=prepared.deterministic,
        enable_tf32=prepared.enable_tf32,
    )
    return prepared, device


def _build_model(config: TrainingConfig, device: torch.device) -> ScratchCNN:
    model = ScratchCNN()
    if config.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model.to(device)


def run_training(config: TrainingConfig) -> RunArtifacts:
    """Train the required CNN and persist every artifact needed by the report."""

    config, device = _prepare_runtime_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = config.output_dir / "checkpoints"
    plots_dir = config.output_dir / "plots"
    artifacts_dir = config.output_dir / "artifacts"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_data_bundle(config)
    train_counts = bundle.class_counts["train"]
    print(
        "Classes: "
        f"train neg={train_counts['negative']}, pos={train_counts['positive']} | "
        f"val neg={bundle.class_counts['val']['negative']}, pos={bundle.class_counts['val']['positive']} | "
        f"test neg={bundle.class_counts['test']['negative']}, pos={bundle.class_counts['test']['positive']}"
    )
    updates_per_epoch = len(bundle.train_loader)
    if updates_per_epoch < 8:
        print(
            f"Aviso: batch_size={config.batch_size} gera somente {updates_per_epoch} atualizações por época. "
            "Para um dataset pequeno, use batch_size=16 ou 32; VRAM ociosa não aprende por osmose."
        )

    model = _build_model(config, device)
    evaluation_criterion = nn.BCEWithLogitsLoss()
    training_criterion, class_balance = _make_training_criterion(bundle, config, device)
    optimizer, optimizer_name = _make_optimizer(model, config, device)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )
    amp_enabled = bool(config.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if amp_enabled else None

    write_json(
        artifacts_dir / "experiment_config.json",
        {
            "config": config.as_dict(),
            "dataset": {
                "root": str(bundle.dataset_root),
                "split_dirs": {name: str(path) for name, path in bundle.split_dirs.items()},
                "sizes": bundle.sizes,
                "class_names": list(bundle.class_names),
                "class_counts": bundle.class_counts,
            },
            "class_balance": class_balance,
            "runtime": runtime_snapshot(device),
            "optimizer": optimizer_name,
            "model": {
                "name": "ScratchCNN",
                "trainable_parameters": count_trainable_parameters(model),
                "architecture": str(model),
            },
        },
    )

    rows: list[dict[str, float | int | None]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    best_checkpoint = checkpoints_dir / "best_val_loss.pt"
    last_checkpoint = checkpoints_dir / "last.pt"
    _synchronize(device)
    started_at = perf_counter()

    for epoch in range(1, config.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _synchronize(device)
        epoch_started_at = perf_counter()

        train_result = _run_loader(
            model=model,
            loader=bundle.train_loader,
            criterion=training_criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            max_grad_norm=config.max_grad_norm,
            channels_last=config.channels_last,
        )
        val_result = _run_loader(
            model=model,
            loader=bundle.val_loader,
            criterion=evaluation_criterion,
            device=device,
            amp_enabled=amp_enabled,
            channels_last=config.channels_last,
        )
        _synchronize(device)
        epoch_time_seconds = perf_counter() - epoch_started_at
        max_gpu_memory_mb = (
            round(torch.cuda.max_memory_allocated(device) / (1024**2), 1)
            if device.type == "cuda"
            else None
        )

        scheduler.step(val_result.loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        rows.append(
            _history_row(
                epoch,
                train_result,
                val_result,
                current_lr,
                epoch_time_seconds,
                max_gpu_memory_mb,
            )
        )

        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            bundle=bundle,
            epoch=epoch,
            best_val_loss=best_val_loss,
            val_result=val_result,
        )
        atomic_torch_save(last_checkpoint, payload)

        if val_result.loss < best_val_loss:
            best_val_loss = val_result.loss
            best_epoch = epoch
            stale_epochs = 0
            payload["best_val_loss"] = best_val_loss
            atomic_torch_save(best_checkpoint, payload)
        else:
            stale_epochs += 1

        memory_suffix = f" | peak_vram={max_gpu_memory_mb:.0f} MiB" if max_gpu_memory_mb is not None else ""
        print(
            f"Época {epoch:03d}/{config.epochs} | "
            f"train loss={train_result.loss:.4f}, acc={train_result.accuracy:.3f} | "
            f"val loss={val_result.loss:.4f}, acc={val_result.accuracy:.3f} | "
            f"lr={current_lr:.2e} | {epoch_time_seconds:.1f}s{memory_suffix}"
        )
        if stale_epochs >= config.early_stopping_patience:
            print(
                f"Early stopping após {stale_epochs} épocas sem melhorar a loss de validação. "
                f"Melhor época: {best_epoch}."
            )
            break

    _synchronize(device)
    elapsed_seconds = perf_counter() - started_at
    history = pd.DataFrame(rows)
    history_path = artifacts_dir / "history.csv"
    history.to_csv(history_path, index=False)
    plot_learning_curves(history, plots_dir / "learning_curves.png")

    # This checkpoint is produced by this program. Do not load arbitrary untrusted .pt files.
    best_state = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model_state_dict"])
    test_result = _run_loader(
        model=model,
        loader=bundle.test_loader,
        criterion=evaluation_criterion,
        device=device,
        amp_enabled=amp_enabled,
        channels_last=config.channels_last,
    )
    plot_confusion_matrix(test_result, bundle.class_names, plots_dir / "confusion_matrix_test.png")

    prediction_frame = pd.DataFrame(
        {
            "target": test_result.targets,
            "probability_positive": test_result.probabilities,
            "prediction": [int(probability >= 0.5) for probability in test_result.probabilities],
        }
    )
    prediction_frame.to_csv(artifacts_dir / "test_predictions.csv", index=False)

    summary = {
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "training_time_seconds": elapsed_seconds,
        "best_validation_loss": best_val_loss,
        "test_metrics": test_result.compact_dict(),
        "dataset": {
            "root": str(bundle.dataset_root),
            "sizes": bundle.sizes,
            "class_names": list(bundle.class_names),
            "class_counts": bundle.class_counts,
        },
        "class_balance": class_balance,
        "runtime": runtime_snapshot(device),
        "artifacts": {
            "best_checkpoint": str(best_checkpoint),
            "last_checkpoint": str(last_checkpoint),
            "history": str(history_path),
            "learning_curves": str(plots_dir / "learning_curves.png"),
            "confusion_matrix": str(plots_dir / "confusion_matrix_test.png"),
            "test_predictions": str(artifacts_dir / "test_predictions.csv"),
        },
    }
    summary_path = artifacts_dir / "run_summary.json"
    write_json(summary_path, summary)
    return RunArtifacts(
        output_dir=config.output_dir,
        best_checkpoint=best_checkpoint,
        history_path=history_path,
        summary_path=summary_path,
        test_result=test_result,
    )


def evaluate_checkpoint(
    *,
    checkpoint_path: Path,
    data_dir: Path,
    output_dir: Path,
    batch_size: int | None = None,
    num_workers: int | str | None = None,
    device_name: str = "auto",
) -> EvaluationResult:
    """Evaluate a saved CNN on the untouched test split."""

    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    stored_config = TrainingConfig.from_dict(checkpoint["config"])
    config = TrainingConfig(
        **{
            **stored_config.as_dict(),
            "data_dir": data_dir,
            "output_dir": output_dir,
            "batch_size": batch_size or stored_config.batch_size,
            "num_workers": num_workers if num_workers is not None else stored_config.num_workers,
            "device": str(device),
        }
    )
    config = replace(config, num_workers=resolve_num_workers(config.num_workers, device))
    config.validate()
    bundle = build_data_bundle(config)
    model = _build_model(config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    result = _run_loader(
        model=model,
        loader=bundle.test_loader,
        criterion=nn.BCEWithLogitsLoss(),
        device=device,
        amp_enabled=bool(config.use_amp and device.type == "cuda"),
        channels_last=config.channels_last,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "evaluation_summary.json", result.compact_dict())
    plot_confusion_matrix(result, bundle.class_names, output_dir / "confusion_matrix_test.png")
    return result
