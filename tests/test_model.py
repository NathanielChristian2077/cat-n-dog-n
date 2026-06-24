import pytest
import torch

from cnn_cats_dogs.config import PHASE1_RANKED_PRESETS, get_classifier_preset
from cnn_cats_dogs.model import ScratchCNN, count_trainable_parameters


@pytest.mark.parametrize("architecture", PHASE1_RANKED_PRESETS)
def test_phase1_derived_head_accepts_rgb_224_and_has_expected_output(architecture: str) -> None:
    preset = get_classifier_preset(architecture)
    model = ScratchCNN(
        hidden_layers=preset.hidden_layers,
        output_mode=preset.output_mode,
        classifier_dropout=0.10,
    )
    model.eval()
    with torch.inference_mode():
        logits = model(torch.randn(2, 3, 224, 224))
    expected_output_features = 1 if preset.output_mode == "sigmoid1" else 2
    assert logits.shape == (2, expected_output_features)
    assert count_trainable_parameters(model) > 0


def test_rank_one_head_preserves_stage1_topology() -> None:
    preset = get_classifier_preset("phase1_rank1_32x64x512_softmax2")
    assert preset.hidden_layers == (32, 64, 512)
    assert preset.output_mode == "softmax2"
