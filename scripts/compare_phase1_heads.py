#!/usr/bin/env python3
"""Compare Stage 1-derived dense heads without requiring editable installation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cnn_cats_dogs.compare import main


if __name__ == "__main__":
    main()
