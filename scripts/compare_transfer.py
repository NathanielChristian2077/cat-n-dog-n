#!/usr/bin/env python3
"""Run Stage 3 transfer-learning comparison without requiring editable installation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cnn_cats_dogs.transfer_compare import main


if __name__ == "__main__":
    main()
