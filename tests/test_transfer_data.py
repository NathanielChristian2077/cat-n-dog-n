import pytest
from PIL import Image

from cnn_cats_dogs.transfer_data import build_transfer_transforms
from cnn_cats_dogs.transfer_models import transfer_model_ids


@pytest.mark.parametrize("architecture", transfer_model_ids())
def test_transfer_transforms_produce_normalized_rgb_224_tensors(architecture: str) -> None:
    transforms = build_transfer_transforms(architecture)
    image = Image.new("RGB", (320, 280), color=(120, 80, 50))

    train_tensor = transforms.train(image)
    eval_tensor = transforms.evaluation(image)

    assert train_tensor.shape == (3, 224, 224)
    assert eval_tensor.shape == (3, 224, 224)
    assert len(transforms.metadata["train_augmentations"]) >= 3
