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


def set_global_seed(seed: int, *, deterministic: bool, enable_tf32: bool) -> None:
    """Seed RNGs and configure the CUDA execution policy explicitly.

    Deterministic kernels and maximum throughput are competing objectives. The
    default project profile favours a fast local CUDA run; ``--deterministic``
    is retained for a controlled rerun rather than penalising every experiment.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    if torch.cuda.is_available():
        # RTX-class GPUs benefit from TF32 for float32 matrix/convolution paths.
        # AMP remains the main acceleration path; TF32 is a safe fast fallback.
        torch.backends.cuda.matmul.allow_tf32 = enable_tf32 and not deterministic
        torch.backends.cudnn.allow_tf32 = enable_tf32 and not deterministic
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if enable_tf32 and not deterministic else "highest")


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy inside each DataLoader worker.

    PyTorch derives a distinct worker seed from the generator passed to the
    loader. Keeping this helper local makes randomized image augmentations
    reproducible when a controlled run is requested.
    """

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def resolve_num_workers(requested: int | str, device: torch.device) -> int:
    """Choose a conservative loader-worker count for the available host CPU.

    More workers are not automatically faster. Eight workers keep image decode
    and augmentation ahead of a single local GPU without turning WSL/desktop
    scheduling into a small civil war.
    """

    if isinstance(requested, int):
        if requested < 0:
            raise ValueError("num_workers precisa ser não negativo.")
        return requested
    if requested != "auto":
        raise ValueError("num_workers precisa ser um inteiro não negativo ou 'auto'.")

    logical_cpus = os.cpu_count() or 2
    if device.type == "cuda":
        return min(8, max(2, logical_cpus - 2))
    return min(4, max(0, logical_cpus - 1))


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
        "logical_cpu_count": os.cpu_count(),
        "pytorch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        snapshot.update(
            {
                "gpu_name": properties.name,
                "gpu_memory_gb": round(properties.total_memory / (1024**3), 2),
                "gpu_compute_capability": f"{properties.major}.{properties.minor}",
                "cudnn_version": torch.backends.cudnn.version(),
                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            }
        )
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
