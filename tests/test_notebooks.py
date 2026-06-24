import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("notebooks/etapa2_cnn_pytorch.ipynb"),
        Path("notebooks/etapa3_transfer_learning.ipynb"),
    ],
)
def test_delivery_notebook_is_valid_json_and_has_cells(relative_path: Path) -> None:
    notebook_path = ROOT / relative_path
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert payload["nbformat"] == 4
    assert payload["cells"]
    assert any(cell["cell_type"] == "code" for cell in payload["cells"])
    assert any(cell["cell_type"] == "markdown" for cell in payload["cells"])
