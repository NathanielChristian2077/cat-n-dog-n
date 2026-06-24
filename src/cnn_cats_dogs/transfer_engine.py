"""Two-phase transfer-learning training, checkpointing, and evaluation."""

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

from .metrics import EvaluationResult, make_evaluation_result
from .transfer_config import TransferTrainingConfig
from .transfer_data import TransferDataBundle, build_transfer_data_bundle
from .transfer_models import (
    TransferPhase,
    configure_trainability,
    create_transfer_model,
    get_transfer_model_spec,
    set_frozen_batch_norm_eval,
    transfer_model_metadata,
)
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
class TransferRunArtifacts:
    """Stable paths emitted by one two-phase transfer-learning run."""

    output_dir: Path
    best_checkpoint: Path
    phase1_checkpoint: Path
    phase2_checkpoint: Path
    history_path: Path
    summary_path: Path
    test_result: EvaluationResult | None


@dataclass
class _SelectionState:
    """Mutable validation-selection state shared by both optimisation phases."""

    best_val_loss: float = float("inf")
    best_phase: str | None = None
    best_global_epoch: int = 0


@dataclass(frozen=True)
class _PhaseOutcome:
    phase: TransferPhase
    best_checkpoint: Path
    best_val_loss: float
    best_phase_epoch: int
    completed_epochs: int
    trainability: dict[str, int | str]


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
    return images, labels.to(device, non_blocking=non_blocking).long()


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
    """Run one full loader and collect probability-based binary metrics."""

    is_training = optimizer is not None
    model.train(is_training)
    if is_training:
        # Frozen ResNet BatchNorm layers must not rewrite ImageNet running stats.
        set_frozen_batch_norm_eval(model)

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
                if logits.ndim != 2 or logits.shape[1] != 2:
                    raise RuntimeError(
                        f"Transfer learning requer dois logits por imagem; recebido shape={tuple(logits.shape)}."
                    )
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
        all_targets.extend(labels.detach().cpu().tolist())
        all_probabilities.extend(torch.softmax(logits.detach(), dim=1)[:, 1].float().cpu().tolist())

    return make_evaluation_result(
        mean_loss=total_loss / max(total_samples, 1),
        targets=all_targets,
        probabilities=all_probabilities,
    )


def _make_training_criterion(
    bundle: TransferDataBundle,
    config: TransferTrainingConfig,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Use class-weighted CrossEntropy only when the supplied train split needs it."""

    counts = bundle.class_counts["train"]
    positive_weight = counts["negative"] / counts["positive"]
    if not config.balance_positive_class or abs(positive_weight - 1.0) < 1e-12:
        return nn.CrossEntropyLoss(), {
            "enabled": False,
            "class_weights": [1.0, 1.0],
            "loss": "CrossEntropyLoss",
        }

    weights = torch.tensor([1.0, positive_weight], device=device, dtype=torch.float32)
    return nn.CrossEntropyLoss(weight=weights), {
        "enabled": True,
        "class_weights": [1.0, float(positive_weight)],
        "loss": "CrossEntropyLoss(class_weight)",
    }


def _make_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[AdamW, str]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("Nenhum parâmetro treinável foi configurado para a fase atual.")

    kwargs = {"lr": learning_rate, "weight_decay": weight_decay}
    if device.type == "cuda":
        try:
            return AdamW(trainable, fused=True, **kwargs), "AdamW(fused=True)"
        except (RuntimeError, TypeError):
            pass
    return AdamW(trainable, **kwargs), "AdamW"


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: ReduceLROnPlateau,
    config: TransferTrainingConfig,
    bundle: TransferDataBundle,
    phase: TransferPhase,
    phase_epoch: int,
    global_epoch: int,
    phase_best_val_loss: float,
    val_result: EvaluationResult,
    trainability: dict[str, int | str],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "kind": "transfer_learning_checkpoint",
        "architecture": transfer_model_metadata(config.architecture),
        "phase": phase,
        "phase_epoch": phase_epoch,
        "global_epoch": global_epoch,
        "phase_best_val_loss": phase_best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config.as_dict(),
        "class_names": list(bundle.class_names),
        "positive_class": bundle.positive_class,
        "class_counts": bundle.class_counts,
        "dataset_root": str(bundle.dataset_root),
        "split_dirs": {name: str(path) for name, path in bundle.split_dirs.items()},
        "transforms": bundle.transform_metadata,
        "trainability": trainability,
        "validation_metrics": val_result.compact_dict(),
    }


def _history_row(
    *,
    phase: TransferPhase,
    phase_epoch: int,
    global_epoch: int,
    train: EvaluationResult,
    train_clean: EvaluationResult,
    validation: EvaluationResult,
    learning_rate: float,
    epoch_time_seconds: float,
    max_gpu_memory_mb: float | None,
    trainable_parameters: int,
) -> dict[str, float | int | str | None]:
    return {
        "phase": phase,
        "phase_epoch": phase_epoch,
        "epoch": global_epoch,
        "learning_rate": learning_rate,
        "train_loss": train.loss,
        "train_accuracy": train.accuracy,
        "train_precision": train.precision,
        "train_recall": train.recall,
        "train_f1": train.f1,
        "train_clean_loss": train_clean.loss,
        "train_clean_accuracy": train_clean.accuracy,
        "train_clean_precision": train_clean.precision,
        "train_clean_recall": train_clean.recall,
        "train_clean_f1": train_clean.f1,
        "val_loss": validation.loss,
        "val_accuracy": validation.accuracy,
        "val_precision": validation.precision,
        "val_recall": validation.recall,
        "val_f1": validation.f1,
        "epoch_time_seconds": epoch_time_seconds,
        "max_gpu_memory_mb": max_gpu_memory_mb,
        "trainable_parameters": trainable_parameters,
    }


def _prepare_runtime_config(config: TransferTrainingConfig) -> tuple[TransferTrainingConfig, torch.device]:
    device = resolve_device(config.device)
    workers = resolve_num_workers(config.num_workers, device)
    prepared = replace(config, device=str(device), num_workers=workers)
    prepared.validate()
    set_global_seed(
        prepared.seed,
        deterministic=prepared.deterministic,
        enable_tf32=prepared.enable_tf32,
    )
    return prepared, device


def _build_model(config: TransferTrainingConfig, device: torch.device, *, pretrained: bool) -> nn.Module:
    model = create_transfer_model(config.architecture, pretrained=pretrained)
    if config.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model.to(device)


def _run_phase(
    *,
    phase: TransferPhase,
    phase_epochs: int,
    learning_rate: float,
    model: nn.Module,
    config: TransferTrainingConfig,
    bundle: TransferDataBundle,
    device: torch.device,
    output_dir: Path,
    state: _SelectionState,
    history_rows: list[dict[str, float | int | str | None]],
    global_epoch_offset: int,
) -> _PhaseOutcome:
    """Train one controlled phase and update global validation selection."""

    checkpoints_dir = output_dir / "checkpoints"
    phase_best_checkpoint = checkpoints_dir / f"{phase}_best_val_loss.pt"
    global_best_checkpoint = checkpoints_dir / "best_val_loss.pt"
    last_checkpoint = checkpoints_dir / "last.pt"

    trainability = configure_trainability(model, config.architecture, phase)
    optimizer, optimizer_name = _make_optimizer(model, learning_rate, config.weight_decay, device)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )
    training_criterion, _ = _make_training_criterion(bundle, config, device)
    evaluation_criterion = nn.CrossEntropyLoss()
    amp_enabled = bool(config.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if amp_enabled else None

    phase_best_loss = float("inf")
    phase_best_epoch = 0
    stale_epochs = 0
    completed_epochs = 0

    print(
        f"\nFase {phase}: epochs={phase_epochs}, lr={learning_rate:.2e}, "
        f"trainable={trainability['trainable_parameters']:,}/{trainability['total_parameters']:,} "
        f"({trainability['trainable_modules']}) | optimizer={optimizer_name}"
    )

    for phase_epoch in range(1, phase_epochs + 1):
        completed_epochs = phase_epoch
        global_epoch = global_epoch_offset + phase_epoch
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _synchronize(device)
        started_at = perf_counter()

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
        train_clean_result = _run_loader(
            model=model,
            loader=bundle.train_eval_loader,
            criterion=evaluation_criterion,
            device=device,
            amp_enabled=amp_enabled,
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
        epoch_seconds = perf_counter() - started_at
        max_gpu_memory_mb = (
            round(torch.cuda.max_memory_allocated(device) / (1024**2), 1)
            if device.type == "cuda"
            else None
        )

        scheduler.step(val_result.loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history_rows.append(
            _history_row(
                phase=phase,
                phase_epoch=phase_epoch,
                global_epoch=global_epoch,
                train=train_result,
                train_clean=train_clean_result,
                validation=val_result,
                learning_rate=current_lr,
                epoch_time_seconds=epoch_seconds,
                max_gpu_memory_mb=max_gpu_memory_mb,
                trainable_parameters=int(trainability["trainable_parameters"]),
            )
        )

        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            bundle=bundle,
            phase=phase,
            phase_epoch=phase_epoch,
            global_epoch=global_epoch,
            phase_best_val_loss=phase_best_loss,
            val_result=val_result,
            trainability=trainability,
        )
        atomic_torch_save(last_checkpoint, payload)

        improved_phase = val_result.loss < phase_best_loss
        if improved_phase:
            phase_best_loss = val_result.loss
            phase_best_epoch = phase_epoch
            stale_epochs = 0
            payload["phase_best_val_loss"] = phase_best_loss
            atomic_torch_save(phase_best_checkpoint, payload)
        else:
            stale_epochs += 1

        if val_result.loss < state.best_val_loss:
            state.best_val_loss = val_result.loss
            state.best_phase = phase
            state.best_global_epoch = global_epoch
            atomic_torch_save(global_best_checkpoint, payload)

        memory_suffix = f" | peak_vram={max_gpu_memory_mb:.0f} MiB" if max_gpu_memory_mb is not None else ""
        print(
            f"{phase} {phase_epoch:03d}/{phase_epochs} | "
            f"train_aug={train_result.accuracy:.3f} | train_clean={train_clean_result.accuracy:.3f} | "
            f"val loss={val_result.loss:.4f}, acc={val_result.accuracy:.3f} | "
            f"lr={current_lr:.2e} | {epoch_seconds:.1f}s{memory_suffix}"
        )
        if stale_epochs >= config.early_stopping_patience:
            print(
                f"Early stopping na fase {phase} após {stale_epochs} épocas sem melhorar a validação. "
                f"Melhor época da fase: {phase_best_epoch}."
            )
            break

    return _PhaseOutcome(
        phase=phase,
        best_checkpoint=phase_best_checkpoint,
        best_val_loss=phase_best_loss,
        best_phase_epoch=phase_best_epoch,
        completed_epochs=completed_epochs,
        trainability=trainability,
    )


def _save_test_artifacts(
    *,
    result: EvaluationResult,
    class_names: tuple[str, str],
    output_dir: Path,
) -> None:
    plots_dir = output_dir / "plots"
    artifacts_dir = output_dir / "artifacts"
    plot_confusion_matrix(result, class_names, plots_dir / "confusion_matrix_test.png")
    pd.DataFrame(
        {
            "target": result.targets,
            "probability_positive": result.probabilities,
            "prediction": [int(probability >= 0.5) for probability in result.probabilities],
        }
    ).to_csv(artifacts_dir / "test_predictions.csv", index=False)


def run_transfer_training(config: TransferTrainingConfig) -> TransferRunArtifacts:
    """Run frozen-head adaptation followed by partial fine-tuning.

    Test evaluation is optional by design. During architecture selection, set
    ``evaluate_test=False`` and choose by validation loss. Evaluate the selected
    checkpoint once later with ``evaluate_transfer_checkpoint``.
    """

    config, device = _prepare_runtime_config(config)
    spec = get_transfer_model_spec(config.architecture)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (config.output_dir / "plots").mkdir(parents=True, exist_ok=True)
    (config.output_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    bundle = build_transfer_data_bundle(config)
    print(f"Modelo: {spec.display_name} | pesos={type(spec.weights).__name__}.{spec.weights.name}")
    print(
        "Classes: "
        f"train neg={bundle.class_counts['train']['negative']}, pos={bundle.class_counts['train']['positive']} | "
        f"val neg={bundle.class_counts['val']['negative']}, pos={bundle.class_counts['val']['positive']} | "
        f"test neg={bundle.class_counts['test']['negative']}, pos={bundle.class_counts['test']['positive']}"
    )

    model = _build_model(config, device, pretrained=True)
    initial_trainability = configure_trainability(model, config.architecture, "head")
    write_json(
        config.output_dir / "artifacts" / "experiment_config.json",
        {
            "config": config.as_dict(),
            "architecture": transfer_model_metadata(config.architecture),
            "dataset": {
                "root": str(bundle.dataset_root),
                "split_dirs": {name: str(path) for name, path in bundle.split_dirs.items()},
                "sizes": bundle.sizes,
                "class_names": list(bundle.class_names),
                "class_counts": bundle.class_counts,
            },
            "transforms": bundle.transform_metadata,
            "runtime": runtime_snapshot(device),
            "initial_head_trainability": initial_trainability,
        },
    )

    selection = _SelectionState()
    history_rows: list[dict[str, float | int | str | None]] = []
    _synchronize(device)
    started_at = perf_counter()

    phase1 = _run_phase(
        phase="head",
        phase_epochs=config.head_epochs,
        learning_rate=config.head_learning_rate,
        model=model,
        config=config,
        bundle=bundle,
        device=device,
        output_dir=config.output_dir,
        state=selection,
        history_rows=history_rows,
        global_epoch_offset=0,
    )

    # Start partial fine-tuning from the best head-adaptation checkpoint, not
    # merely the last epoch that happened to run before early stopping.
    phase1_state = torch.load(phase1.best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(phase1_state["model_state_dict"])
    phase2 = _run_phase(
        phase="finetune",
        phase_epochs=config.finetune_epochs,
        learning_rate=config.finetune_learning_rate,
        model=model,
        config=config,
        bundle=bundle,
        device=device,
        output_dir=config.output_dir,
        state=selection,
        history_rows=history_rows,
        global_epoch_offset=phase1.completed_epochs,
    )

    _synchronize(device)
    elapsed_seconds = perf_counter() - started_at
    history = pd.DataFrame(history_rows)
    history_path = config.output_dir / "artifacts" / "history.csv"
    history.to_csv(history_path, index=False)
    plot_learning_curves(history, config.output_dir / "plots" / "learning_curves.png")

    best_checkpoint = config.output_dir / "checkpoints" / "best_val_loss.pt"
    test_result: EvaluationResult | None = None
    if config.evaluate_test:
        best_state = torch.load(best_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(best_state["model_state_dict"])
        test_result = _run_loader(
            model=model,
            loader=bundle.test_loader,
            criterion=nn.CrossEntropyLoss(),
            device=device,
            amp_enabled=bool(config.use_amp and device.type == "cuda"),
            channels_last=config.channels_last,
        )
        _save_test_artifacts(result=test_result, class_names=bundle.class_names, output_dir=config.output_dir)

    summary = {
        "architecture": transfer_model_metadata(config.architecture),
        "selected_checkpoint": {
            "path": str(best_checkpoint),
            "phase": selection.best_phase,
            "global_epoch": selection.best_global_epoch,
            "best_validation_loss": selection.best_val_loss,
        },
        "phases": {
            "head": {
                "best_checkpoint": str(phase1.best_checkpoint),
                "best_validation_loss": phase1.best_val_loss,
                "best_phase_epoch": phase1.best_phase_epoch,
                "completed_epochs": phase1.completed_epochs,
                "trainability": phase1.trainability,
            },
            "finetune": {
                "best_checkpoint": str(phase2.best_checkpoint),
                "best_validation_loss": phase2.best_val_loss,
                "best_phase_epoch": phase2.best_phase_epoch,
                "completed_epochs": phase2.completed_epochs,
                "trainability": phase2.trainability,
            },
        },
        "training_time_seconds": elapsed_seconds,
        "test_evaluated": config.evaluate_test,
        "test_metrics": test_result.compact_dict() if test_result is not None else None,
        "dataset": {
            "root": str(bundle.dataset_root),
            "sizes": bundle.sizes,
            "class_names": list(bundle.class_names),
            "class_counts": bundle.class_counts,
        },
        "transforms": bundle.transform_metadata,
        "runtime": runtime_snapshot(device),
        "artifacts": {
            "best_checkpoint": str(best_checkpoint),
            "phase1_checkpoint": str(phase1.best_checkpoint),
            "phase2_checkpoint": str(phase2.best_checkpoint),
            "history": str(history_path),
            "learning_curves": str(config.output_dir / "plots" / "learning_curves.png"),
            "confusion_matrix": str(config.output_dir / "plots" / "confusion_matrix_test.png"),
            "test_predictions": str(config.output_dir / "artifacts" / "test_predictions.csv"),
        },
    }
    summary_path = config.output_dir / "artifacts" / "run_summary.json"
    write_json(summary_path, summary)

    return TransferRunArtifacts(
        output_dir=config.output_dir,
        best_checkpoint=best_checkpoint,
        phase1_checkpoint=phase1.best_checkpoint,
        phase2_checkpoint=phase2.best_checkpoint,
        history_path=history_path,
        summary_path=summary_path,
        test_result=test_result,
    )


def evaluate_transfer_checkpoint(
    *,
    checkpoint_path: Path,
    data_dir: Path,
    output_dir: Path,
    batch_size: int | None = None,
    num_workers: int | str | None = None,
    device_name: str = "auto",
) -> EvaluationResult:
    """Evaluate a chosen transfer-learning checkpoint once on the fixed test split."""

    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("kind") != "transfer_learning_checkpoint":
        raise ValueError("O checkpoint informado não foi produzido pelo motor de transfer learning.")

    stored_config = TransferTrainingConfig.from_dict(checkpoint["config"])
    config = replace(
        stored_config,
        data_dir=data_dir,
        output_dir=output_dir,
        batch_size=batch_size or stored_config.batch_size,
        num_workers=num_workers if num_workers is not None else stored_config.num_workers,
        device=str(device),
        evaluate_test=True,
    )
    config = replace(config, num_workers=resolve_num_workers(config.num_workers, device))
    config.validate()
    bundle = build_transfer_data_bundle(config)
    model = _build_model(config, device, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    result = _run_loader(
        model=model,
        loader=bundle.test_loader,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        amp_enabled=bool(config.use_amp and device.type == "cuda"),
        channels_last=config.channels_last,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    _save_test_artifacts(result=result, class_names=bundle.class_names, output_dir=output_dir)
    write_json(
        output_dir / "artifacts" / "evaluation_summary.json",
        {
            "checkpoint": str(checkpoint_path),
            "architecture": transfer_model_metadata(config.architecture),
            "test_metrics": result.compact_dict(),
            "dataset": {
                "root": str(bundle.dataset_root),
                "sizes": bundle.sizes,
                "class_names": list(bundle.class_names),
                "class_counts": bundle.class_counts,
            },
            "transforms": bundle.transform_metadata,
            "runtime": runtime_snapshot(device),
        },
    )
    return result
