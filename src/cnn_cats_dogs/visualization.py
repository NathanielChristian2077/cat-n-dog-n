"""Plots saved for the notebook and final report."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import EvaluationResult


def plot_learning_curves(history: pd.DataFrame, output_path: Path) -> None:
    """Plot loss and accuracy by epoch without requiring the notebook to stay open."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history["epoch"], history["train_loss"], label="Treino")
    axes[0].plot(history["epoch"], history["val_loss"], label="Validação")
    axes[0].set_title("Loss por época")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Binary Cross-Entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(history["epoch"], history["train_accuracy"], label="Treino")
    axes[1].plot(history["epoch"], history["val_accuracy"], label="Validação")
    axes[1].set_title("Acurácia por época")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Acurácia")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    result: EvaluationResult,
    class_names: Sequence[str],
    output_path: Path,
) -> None:
    """Save a labeled 2x2 confusion-matrix image for the report."""

    if len(class_names) != 2:
        raise ValueError("A matriz de confusão binária requer dois nomes de classe.")

    matrix = np.asarray(result.confusion_matrix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    image = ax.imshow(matrix)
    fig.colorbar(image, ax=ax)

    ax.set_xticks([0, 1], labels=class_names)
    ax.set_yticks([0, 1], labels=class_names)
    ax.set_xlabel("Classe predita")
    ax.set_ylabel("Classe real")
    ax.set_title("Matriz de confusão - conjunto de teste")

    threshold = matrix.max() / 2 if matrix.max() else 0
    for row in range(2):
        for col in range(2):
            color = "white" if matrix[row, col] > threshold else "black"
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
