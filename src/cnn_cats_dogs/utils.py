"""Reproducibility, runtime selection, and safe artifact-writing helpers."""

from __future__ import annotations

import json
import os
import platform
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch as far as practical.

    Determinism can reduce throughput on CUDA. Here reproducibility matters more
    than squeezing an extra fraction of a second out of a class assignment.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed DataLoader workers from PyTorch's worker-specific initial seed."""

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def resolve_device(requested: str) -> torch.device:
    """Resolve auto/cpu/cuda/mps and fail loudly for unavailable explicit devices."""

    requested = requested.lower().strip()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA foi solicitada, mas torch.cuda.is_available() retornou False.")
    if requested == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS foi solicitada, mas não está disponível neste ambiente.")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device precisa ser auto, cpu, cuda ou mps.")
    return torch.device(requested)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def runtime_snapshot(device: torch.device) -> dict[str, Any]:
    """Capture environment information that belongs in a reproducible report."""

    snapshot: dict[str, Any] = {
        "timestamp_utc": utc_now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        snapshot["gpu_name"] = properties.name
        snapshot["gpu_memory_gb"] = round(properties.total_memory / (1024**3), 2)
        snapshot["cudnn_version"] = torch.backends.cudnn.version()
    return snapshot


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Objeto não serializável em JSON: {type(value)!r}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically so interrupted jobs do not leave a half-file behind."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as tmp:
        json.dump(payload, tmp, indent=2, ensure_ascii=False, default=_json_default)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    """Save checkpoints atomically; a killed notebook should not corrupt the best model."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".tmp") as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
