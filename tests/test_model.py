import torch

from cnn_cats_dogs.model import ScratchCNN, count_trainable_parameters


def test_scratch_cnn_accepts_rgb_224_and_returns_one_logit_per_image() -> None:
    model = ScratchCNN()
    model.eval()
    with torch.inference_mode():
        logits = model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 1)
    assert count_trainable_parameters(model) > 0
