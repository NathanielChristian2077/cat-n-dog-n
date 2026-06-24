#!/usr/bin/env python3
"""Evaluate a selected Stage 3 transfer checkpoint without editable installation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cnn_cats_dogs.transfer_evaluate import main


if __name__ == "__main__":
    main()
