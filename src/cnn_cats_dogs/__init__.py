"""Reusable components for the IFSC cats-versus-dogs CNN assignment."""

from .config import TrainingConfig
from .model import ScratchCNN

__all__ = ["ScratchCNN", "TrainingConfig"]
