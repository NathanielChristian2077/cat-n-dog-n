"""Binary classification metrics and serializable evaluation records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


@dataclass(frozen=True)
class EvaluationResult:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    n_samples: int
    confusion_matrix: list[list[int]]
    targets: list[int]
    probabilities: list[float]

    def compact_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("targets")
        payload.pop("probabilities")
        return payload


def make_evaluation_result(
    *,
    mean_loss: float,
    targets: list[int],
    probabilities: list[float],
) -> EvaluationResult:
    if not targets:
        raise ValueError("Não é possível calcular métricas sem amostras.")
    target_array = np.asarray(targets, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    predicted_array = (probability_array >= 0.5).astype(np.int64)

    accuracy = float((predicted_array == target_array).mean())
    precision, recall, f1, _ = precision_recall_fscore_support(
        target_array,
        predicted_array,
        average="binary",
        zero_division=0,
    )
    matrix = confusion_matrix(target_array, predicted_array, labels=[0, 1]).astype(int).tolist()

    return EvaluationResult(
        loss=float(mean_loss),
        accuracy=accuracy,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        n_samples=int(target_array.size),
        confusion_matrix=matrix,
        targets=target_array.tolist(),
        probabilities=probability_array.tolist(),
    )
