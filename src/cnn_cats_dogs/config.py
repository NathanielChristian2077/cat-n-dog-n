"""Configuration objects and comparable CNN classifier presets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

OutputMode = Literal["sigmoid1", "softmax2"]


@dataclass(frozen=True)
class ClassifierPreset:
    """Dense head topology inherited from a successful Stage 1 MLP variant."""

    identifier: str
    hidden_layers: tuple[int, ...]
    output_mode: OutputMode
    stage1_rank: int | None
    description: str


CLASSIFIER_PRESETS: dict[str, ClassifierPreset] = {
    "phase1_rank1_32x64x512_softmax2": ClassifierPreset(
        identifier="phase1_rank1_32x64x512_softmax2",
        hidden_layers=(32, 64, 512),
        output_mode="softmax2",
        stage1_rank=1,
        description="Etapa 1 rank 1: 32x64x512 com duas saídas Softmax.",
    ),
    "phase1_rank2_32x64x512_sigmoid1": ClassifierPreset(
        identifier="phase1_rank2_32x64x512_sigmoid1",
        hidden_layers=(32, 64, 512),
        output_mode="sigmoid1",
        stage1_rank=2,
        description="Etapa 1 rank 2: 32x64x512 com uma saída Sigmoid.",
    ),
    "phase1_rank3_64x32x512_sigmoid1": ClassifierPreset(
        identifier="phase1_rank3_64x32x512_sigmoid1",
        hidden_layers=(64, 32, 512),
        output_mode="sigmoid1",
        stage1_rank=3,
        description="Etapa 1 rank 3: 64x32x512 com uma saída Sigmoid.",
    ),
    "phase1_rank4_128x32x512_sigmoid1": ClassifierPreset(
        identifier="phase1_rank4_128x32x512_sigmoid1",
        hidden_layers=(128, 32, 512),
        output_mode="sigmoid1",
        stage1_rank=4,
        description="Etapa 1 rank 4: 128x32x512 com uma saída Sigmoid.",
    ),
    "phase1_rank5_64x32x512_softmax2": ClassifierPreset(
        identifier="phase1_rank5_64x32x512_softmax2",
        hidden_layers=(64, 32, 512),
        output_mode="softmax2",
        stage1_rank=5,
        description="Etapa 1 rank 5: 64x32x512 com duas saídas Softmax.",
    ),
    "phase1_rank6_128x32x512_softmax2": ClassifierPreset(
        identifier="phase1_rank6_128x32x512_softmax2",
        hidden_layers=(128, 32, 512),
        output_mode="softmax2",
        stage1_rank=6,
        description="Etapa 1 rank 6: 128x32x512 com duas saídas Softmax.",
    ),
    "cnn_baseline_128_sigmoid1": ClassifierPreset(
        identifier="cnn_baseline_128_sigmoid1",
        hidden_layers=(128,),
        output_mode="sigmoid1",
        stage1_rank=None,
        description="Baseline CNN anterior: 128 com uma saída Sigmoid.",
    ),
}

PHASE1_RANKED_PRESETS: tuple[str, ...] = tuple(
    preset.identifier
    for preset in sorted(
        (preset for preset in CLASSIFIER_PRESETS.values() if preset.stage1_rank is not None),
        key=lambda preset: preset.stage1_rank or 0,
    )
)


def architecture_ids() -> tuple[str, ...]:
    """Return stable CLI choices, ranked Stage 1 variants first."""

    return (*PHASE1_RANKED_PRESETS, "cnn_baseline_128_sigmoid1")


def get_classifier_preset(identifier: str) -> ClassifierPreset:
    """Resolve an explicit classifier preset or fail before a long CUDA job starts."""

    try:
        return CLASSIFIER_PRESETS[identifier]
    except KeyError as error:
        available = ", ".join(architecture_ids())
        raise ValueError(f"Arquitetura desconhecida: {identifier!r}. Opções: {available}.") from error


@dataclass(slots=True)
class TrainingConfig:
    """All experiment settings in one serializable object."""

    data_dir: Path
    output_dir: Path
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 25
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    prefetch_factor: int = 2
    seed: int = 42
    positive_class: str = "dogs"
    architecture: str = "phase1_rank1_32x64x512_softmax2"
    classifier_dropout: float = 0.10
    device: str = "auto"
    use_amp: bool = True
    deterministic: bool = False
    enable_tf32: bool = True
    channels_last: bool = True
    balance_positive_class: bool = True
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
        if not 0.0 <= self.classifier_dropout < 1.0:
            raise ValueError("classifier_dropout precisa estar no intervalo [0, 1).")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience precisa ser ao menos 1.")
        if self.scheduler_patience < 0:
            raise ValueError("scheduler_patience não pode ser negativo.")
        if not 0 < self.scheduler_factor < 1:
            raise ValueError("scheduler_factor precisa estar entre 0 e 1.")
        if not self.positive_class.strip():
            raise ValueError("positive_class não pode ser vazio.")
        get_classifier_preset(self.architecture)

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
        # Version-3-and-earlier checkpoints did not encode a classifier preset.
        normalized.setdefault("architecture", "cnn_baseline_128_sigmoid1")
        normalized.setdefault("classifier_dropout", 0.10)
        return cls(**normalized)
