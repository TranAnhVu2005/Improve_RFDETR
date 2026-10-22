"""Focused regression tests for RF-DETR Improve v1 P0 integration contracts."""

from pathlib import Path

import pytest
import torch

from train_model.train_improve_v1 import (
    ImprovementConfig,
    _improvement_config,
    _merge_loss_dicts,
    _training_kwargs,
)


def test_improvement_config_uses_rfdetr_classification_loss_key():
    options, weights = _improvement_config(
        {
            "loss_weights": {
                "loss_ce": 1.0,
                "loss_bbox": 5.0,
                "loss_giou": 2.0,
            }
        }
    )
    assert isinstance(options, ImprovementConfig)
    assert weights["loss_ce"] == 1.0
    assert "loss_vfl" not in weights


def test_legacy_loss_vfl_config_alias_remains_supported():
    _, weights = _improvement_config({"loss_weights": {"loss_vfl": 1.25}})
    assert weights["loss_ce"] == 1.25


def test_conflicting_classification_loss_aliases_fail_fast():
    with pytest.raises(ValueError, match="Specify only loss_weights.loss_ce"):
        _improvement_config(
            {"loss_weights": {"loss_vfl": 1.0, "loss_ce": 2.0}}
        )


def test_training_kwargs_pass_encoder_lr_and_checkpoint_interval():
    kwargs = _training_kwargs(
        dataset_dir="/dataset",
        training={
            "batch_size": 16,
            "max_physical_batch_size": 4,
            "grad_accum_steps": 1,
            "lr_encoder": 1.5e-4,
            "checkpoint_interval": 10,
        },
        output_dir=Path("/tmp/output"),
        candidate={"batch_size": 16, "lr": 1e-4, "weight_decay": 1e-4},
        epochs=1,
    )
    assert kwargs["lr_encoder"] == pytest.approx(1.5e-4)
    assert kwargs["checkpoint_interval"] == 10
    assert kwargs["batch_size"] == 4
    assert kwargs["grad_accum_steps"] == 4


def test_merge_preserves_auxiliary_losses_and_replaces_only_final_box_losses():
    base = {
        "loss_ce": torch.tensor(1.0),
        "loss_bbox": torch.tensor(2.0),
        "loss_giou": torch.tensor(3.0),
        "loss_bbox_0": torch.tensor(4.0),
        "loss_giou_0": torch.tensor(5.0),
        "class_error": torch.tensor(0.0),
    }
    bbox_aux = base["loss_bbox_0"]
    giou_aux = base["loss_giou_0"]
    extra = {
        "loss_bbox": torch.tensor(20.0),
        "loss_giou": torch.tensor(30.0),
        "loss_iqcm": torch.tensor(0.1),
    }
    merged = _merge_loss_dicts(
        base,
        extra,
        weight_dict={
            "loss_ce": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_bbox_0": 5.0,
            "loss_giou_0": 2.0,
            "loss_iqcm": 0.25,
        },
    )
    assert merged["loss_bbox"].item() == 20.0
    assert merged["loss_giou"].item() == 30.0
    assert merged["loss_bbox_0"] is bbox_aux
    assert merged["loss_giou_0"] is giou_aux
    assert merged["class_error"].item() == 0.0


def test_merge_fails_if_custom_loss_has_no_weight():
    base = {
        "loss_ce": torch.tensor(1.0),
        "loss_bbox": torch.tensor(2.0),
        "loss_giou": torch.tensor(3.0),
    }
    with pytest.raises(RuntimeError, match="no criterion weights"):
        _merge_loss_dicts(
            base,
            {"loss_iqcm": torch.tensor(0.1)},
            weight_dict={"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0},
        )


def test_improved_yaml_uses_supported_loss_key_and_encoder_lr():
    import yaml

    config_path = (
    Path(__file__).resolve().parent
    / "config"
    / "rfdetr_improve_v1.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["loss_weights"]["loss_ce"] == 1.0
    assert "loss_vfl" not in config["loss_weights"]
    assert config["training"]["lr_encoder"] == pytest.approx(1.5e-4)
    assert config["training"]["checkpoint_interval"] == 10
