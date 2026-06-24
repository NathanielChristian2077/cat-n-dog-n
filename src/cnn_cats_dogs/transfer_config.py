"""Configuration for the Stage 3 transfer-learning experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .transfer_models import get_transfer_model_spec


@dataclass(slots=True)
class TransferTrainingConfig:
    """Serializable settings for one pretrained-backbone experiment.

    The two learning-rate phases are intentionally explicit. A frozen-backbone
    head adaptation and a partial fine-tuning run are different experiments in
    optimisation terms, even when they share one output directory.
    """

    data_dir: Path
    output_dir: Path
    architecture: str
    positive_class: str = "dogs"
    image_size: int = 224
    batch_size: int = 32
    head_epochs: int = 12
    finetune_epochs: int = 20
    head_learning_rate: float = 1e-3
    finetune_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 8
    prefetch_factor: int = 2
    seed: int = 42
    device: str = "auto"
    use_amp: bool = True
    deterministic: bool = False
    enable_tf32: bool = True
    channels_last: bool = True
    balance_positive_class: bool = True
    early_stopping_patience: int = 6
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    max_grad_norm: float = 5.0
    evaluate_test: bool = False

    def validate(self) -> None:
        if self.image_size != 224:
            raise ValueError(
                "A Etapa 3 exige imagens RGB 224x224. Mantenha image_size=224."
            )
        if self.batch_size < 1:
            raise ValueError("batch_size precisa ser positivo.")
        if self.head_epochs < 1:
            raise ValueError("head_epochs precisa ser positivo.")
        if self.finetune_epochs < 1:
            raise ValueError("finetune_epochs precisa ser positivo.")
        if self.head_learning_rate <= 0 or self.finetune_learning_rate <= 0:
            raise ValueError("As learning rates das duas fases precisam ser positivas.")
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
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm precisa ser positivo.")
        if not self.positive_class.strip():
            raise ValueError("positive_class não pode ser vazio.")
        get_transfer_model_spec(self.architecture)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["data_dir"] = str(self.data_dir)
        payload["output_dir"] = str(self.output_dir)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TransferTrainingConfig":
        normalized = dict(payload)
        normalized["data_dir"] = Path(normalized["data_dir"])
        normalized["output_dir"] = Path(normalized["output_dir"])
        normalized.setdefault("evaluate_test", False)
        return cls(**normalized)
