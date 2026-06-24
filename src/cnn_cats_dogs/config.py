"""Configuration objects used by the training and evaluation entry points."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TrainingConfig:
    """All experiment settings in one serializable object.

    The defaults are aimed at a local CUDA run, but every performance-affecting
    choice is retained in the experiment artifact. Reproducibility is available
    through ``deterministic=True`` rather than silently slowing every run.
    """

    data_dir: Path
    output_dir: Path
    image_size: int = 224
    batch_size: int = 128
    epochs: int = 25
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    prefetch_factor: int = 2
    seed: int = 42
    positive_class: str = "dogs"
    device: str = "auto"
    use_amp: bool = True
    deterministic: bool = False
    enable_tf32: bool = True
    channels_last: bool = True
    early_stopping_patience: int = 8
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5
    max_grad_norm: float = 5.0

    def validate(self) -> None:
        if self.image_size != 224:
            raise ValueError(
                "A Etapa 2 exige imagens 224x224. Mantenha image_size=224 para esta entrega."
            )
        if self.batch_size < 1:
            raise ValueError("batch_size precisa ser positivo.")
        if self.epochs < 1:
            raise ValueError("epochs precisa ser positivo.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate precisa ser positivo.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay não pode ser negativo.")
        if self.num_workers < 0:
            raise ValueError("num_workers precisa ser não negativo.")
        if self.prefetch_factor < 1:
            raise ValueError("prefetch_factor precisa ser ao menos 1.")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience precisa ser ao menos 1.")
        if self.scheduler_patience < 0:
            raise ValueError("scheduler_patience não pode ser negativo.")
        if not 0 < self.scheduler_factor < 1:
            raise ValueError("scheduler_factor precisa estar entre 0 e 1.")
        if not self.positive_class.strip():
            raise ValueError("positive_class não pode ser vazio.")

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["data_dir"] = str(self.data_dir)
        payload["output_dir"] = str(self.output_dir)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingConfig":
        normalized = dict(payload)
        normalized["data_dir"] = Path(normalized["data_dir"])
        normalized["output_dir"] = Path(normalized["output_dir"])
        return cls(**normalized)
