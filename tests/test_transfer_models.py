import pytest

from cnn_cats_dogs.transfer_models import (
    configure_trainability,
    create_transfer_model,
    transfer_model_ids,
)


@pytest.mark.parametrize("architecture", transfer_model_ids())
def test_transfer_models_replace_the_final_classifier_with_two_logits(architecture: str) -> None:
    model = create_transfer_model(architecture, pretrained=False)
    if architecture == "resnet18":
        assert model.fc.out_features == 2
    else:
        assert model.classifier[-1].out_features == 2


@pytest.mark.parametrize("architecture", transfer_model_ids())
def test_head_and_partial_finetune_have_distinct_trainable_parameter_sets(architecture: str) -> None:
    model = create_transfer_model(architecture, pretrained=False)
    head = configure_trainability(model, architecture, "head")
    fine_tune = configure_trainability(model, architecture, "finetune")

    assert head["trainable_parameters"] > 0
    assert fine_tune["trainable_parameters"] > head["trainable_parameters"]
    assert fine_tune["total_parameters"] == head["total_parameters"]
