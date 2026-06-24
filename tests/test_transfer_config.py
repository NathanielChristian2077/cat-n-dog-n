from pathlib import Path

from cnn_cats_dogs.transfer_config import TransferTrainingConfig


def test_transfer_config_accepts_selected_model_and_serializes_paths(tmp_path: Path) -> None:
    config = TransferTrainingConfig(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "runs",
        architecture="efficientnet_b0",
    )
    config.validate()
    payload = config.as_dict()

    assert payload["architecture"] == "efficientnet_b0"
    assert payload["data_dir"] == str(tmp_path / "data")
    assert TransferTrainingConfig.from_dict(payload).architecture == "efficientnet_b0"
