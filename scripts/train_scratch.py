#!/usr/bin/env python3
"""Run one configurable Stage 2 scratch-CNN experiment without editable installation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cnn_cats_dogs.cli import main


if __name__ == "__main__":
    main()
