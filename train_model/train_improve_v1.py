"""Train RF-DETR with query calibration, uncertainty, contrastive and IoU losses."""

from __future__ import annotations

import csv
import importlib
import importlib.metadata
import inspect
import json
import math
import random
import shutil
import signal
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from train_model.epoch_logger import EpochLogger


SUPPORTED_RFDETR_VERSION = "1.11.2"
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "rfdetr_improve_v1.yaml"
)


@dataclass(frozen=True)
class ImprovementConfig:
    dqcd_weight: float = 0.08
    iqcm_weight: float = 0.25
    scu_weight: float = 0.1
    lue_weight: float = 0.25
    tversky_weight: float = 0.2
    tversky_alpha: float = 0.7
    tversky_beta: float = 0.3
    hifrm_gamma: float = 2.0
    hard_negative_k: int = 128
    iqcm_top_m: int = 100
    dqcd_start_epoch: int = 8
    dqcd_warmup_epochs: int = 5
    lue_warmup_epochs: int = 15


class ImprovementHeads(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )
        nn.init.zeros_(self.uncertainty_head[-1].weight)
        nn.init.constant_(self.uncertainty_head[-1].bias, -5.0)


def _frequency_gate(samples: Any) -> torch.Tensor:
    """Use high-frequency energy as a proxy; this scope does not add a DAFD module."""
    images = getattr(samples, "tensors", samples)
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise RuntimeError(
            "RF-DETR compatibility error: expected a [batch, channels, height, width] tensor."
        )

    small = F.interpolate(
        images.float(), size=(64, 64), mode="bilinear", align_corners=False
    )
    spectrum = torch.fft.rfft2(small, norm="ortho").abs().mean(dim=1)
    height, width = small.shape[-2:]
    fy = torch.fft.fftfreq(height, device=small.device).abs()[:, None]
    fx = torch.fft.rfftfreq(width, device=small.device)[None, :]
    radius = torch.sqrt(fx.square() + fy.square())
    low = spectrum[:, radius < 0.10].mean(dim=1)
    middle = spectrum[:, (radius >= 0.10) & (radius < 0.25)].mean(dim=1)
    high = spectrum[:, radius >= 0.25].mean(dim=1)
    energy = torch.stack((low, middle, high), dim=1)
    return (energy[:, 2] / energy.sum(dim=1).clamp_min(1e-8)).clamp(0.0, 1.0)


def _box_iou_and_giou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    boxes1_xy = torch.cat((boxes1[..., :2] - boxes1[..., 2:] / 2,
                           boxes1[..., :2] + boxes1[..., 2:] / 2), dim=-1)
    boxes2_xy = torch.cat((boxes2[..., :2] - boxes2[..., 2:] / 2,
                           boxes2[..., :2] + boxes2[..., 2:] / 2), dim=-1)
    left_top = torch.maximum(boxes1_xy[..., :2], boxes2_xy[..., :2])
    right_bottom = torch.minimum(boxes1_xy[..., 2:], boxes2_xy[..., 2:])
    intersection = (right_bottom - left_top).clamp_min(0).prod(dim=-1)
    area1 = (boxes1_xy[..., 2:] - boxes1_xy[..., :2]).clamp_min(0).prod(dim=-1)
    area2 = (boxes2_xy[..., 2:] - boxes2_xy[..., :2]).clamp_min(0).prod(dim=-1)
    union = area1 + area2 - intersection
    iou = intersection / union.clamp_min(1e-8)
    enclosing_left_top = torch.minimum(boxes1_xy[..., :2], boxes2_xy[..., :2])
    enclosing_right_bottom = torch.maximum(
        boxes1_xy[..., 2:], boxes2_xy[..., 2:])
    enclosing_area = (
        (enclosing_right_bottom - enclosing_left_top).clamp_min(0).prod(dim=-1)
    )
    giou = iou - (enclosing_area - union) / enclosing_area.clamp_min(1e-8)
    return iou, giou


def _match_indices(
    criterion: nn.Module,
    outputs: dict[str, Any],
    targets: list[dict[str, torch.Tensor]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    matcher = getattr(criterion, "matcher", None)
    if not callable(matcher):
        raise RuntimeError(
            "RF-DETR compatibility error: criterion.matcher is unavailable.")
    indices = matcher(outputs, targets)
    if len(indices) != len(targets):
        raise RuntimeError(
            "RF-DETR matcher returned an unexpected number of image results.")
    return indices


def _hardness(probabilities: torch.Tensor) -> torch.Tensor:
    objectness = probabilities.max(dim=-1).values
    if probabilities.shape[-1] <= 1:
        return objectness
    return objectness + 0.25 * _ambiguity(probabilities)


def _ambiguity(probabilities: torch.Tensor) -> torch.Tensor:
    if probabilities.shape[-1] <= 1:
        return torch.zeros_like(probabilities[..., 0])
    entropy = -(
        probabilities.clamp(1e-7, 1 - 1e-7)
        * probabilities.clamp(1e-7, 1 - 1e-7).log()
        + (1 - probabilities).clamp(1e-7, 1).mul(
            (1 - probabilities).clamp(1e-7, 1).log()
        )
    ).sum(dim=-1)
    return entropy / math.log(probabilities.shape[-1])


def _hard_negative_indices(
    logits: torch.Tensor,
    matched_queries: torch.Tensor,
    limit: int,
) -> torch.Tensor:
    probabilities = logits.sigmoid()
    candidates = torch.ones(
        logits.shape[0], device=logits.device, dtype=torch.bool
    )
    candidates[matched_queries] = False
    available = candidates.nonzero(as_tuple=False).flatten()
    if available.numel() == 0:
        return available
    k = min(limit, int(available.numel()))
    scores = _hardness(probabilities[available])
    return available[scores.topk(k).indices]


def _custom_losses(
    criterion: nn.Module,
    options: ImprovementConfig,
    outputs: dict[str, Any],
    targets: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    logits = outputs["pred_logits"]
    boxes = outputs["pred_boxes"]
    decoder_states = outputs.get("improve_decoder_states")
    quality = outputs.get("pred_quality")
    quality_logits = outputs.get("pred_quality_logits")
    predicted_log_variance = outputs.get("pred_log_variance")
    gate = outputs.get("improve_frequency_gate")
    if (
        not isinstance(logits, torch.Tensor)
        or not isinstance(boxes, torch.Tensor)
        or not isinstance(decoder_states, torch.Tensor)
        or not isinstance(quality, torch.Tensor)
        or not isinstance(quality_logits, torch.Tensor)
        or not isinstance(predicted_log_variance, torch.Tensor)
        or not isinstance(gate, torch.Tensor)
    ):
        raise RuntimeError("RF-DETR improvement outputs are incomplete.")
    if logits.shape[:2] != boxes.shape[:2] or logits.shape[:2] != quality.shape:
        raise RuntimeError(
            "RF-DETR query dimensions differ between prediction heads.")

    matched = _match_indices(criterion, outputs, targets)
    batch_size, query_count, class_count = logits.shape
    device = logits.device
    matched_mask = torch.zeros(
        (batch_size, query_count), dtype=torch.bool, device=device
    )
    all_predicted: list[torch.Tensor] = []
    all_expected: list[torch.Tensor] = []
    all_quality_predictions: list[torch.Tensor] = []
    all_quality_targets: list[torch.Tensor] = []
    all_uncertainties: list[torch.Tensor] = []
    all_areas: list[torch.Tensor] = []
    giou_losses: list[torch.Tensor] = []
    tversky_losses: list[torch.Tensor] = []
    contrastive_losses: list[torch.Tensor] = []
    normalized_box_count = max(
        1, sum(int(target["boxes"].shape[0]) for target in targets)
    )
    epoch = int(getattr(criterion, "_improve_epoch", 0))
    lue_ramp = min(1.0, (epoch + 1) / options.lue_warmup_epochs)
    dqcd_ramp = min(
        1.0,
        max(0.0, (epoch - options.dqcd_start_epoch + 1) /
            options.dqcd_warmup_epochs),
    )

    for image_index, (source_indices, target_indices) in enumerate(matched):
        source_indices = source_indices.to(device=device, dtype=torch.long)
        target_indices = target_indices.to(device=device, dtype=torch.long)
        matched_mask[image_index, source_indices] = True
        target_boxes = targets[image_index]["boxes"].to(
            device=device, dtype=boxes.dtype)
        target_labels = targets[image_index]["labels"].to(
            device=device, dtype=torch.long)
        image_logits = logits[image_index]
        probabilities = image_logits.sigmoid()

        class_targets = torch.zeros_like(image_logits)
        if source_indices.numel():
            matched_labels = target_labels[target_indices]
            if matched_labels.numel() and (
                int(matched_labels.min()) < 0 or int(
                    matched_labels.max()) >= class_count
            ):
                raise RuntimeError(
                    "Dataset class indices must be within the RF-DETR output range."
                )
            class_targets[source_indices, target_labels[target_indices]] = 1.0

        negative_indices = _hard_negative_indices(
            image_logits, source_indices, options.hard_negative_k
        )
        selected_queries = torch.cat((source_indices, negative_indices))
        if selected_queries.numel():
            selected_probabilities = probabilities[selected_queries]
            selected_targets = class_targets[selected_queries]
            true_positive = (selected_probabilities * selected_targets).sum()
            false_positive = (selected_probabilities *
                              (1 - selected_targets)).sum()
            false_negative = ((1 - selected_probabilities)
                              * selected_targets).sum()
            tversky = true_positive / (
                true_positive
                + options.tversky_alpha * false_positive
                + options.tversky_beta * false_negative
                + 1e-7
            )
            tversky_losses.append(1 - tversky)

        if target_boxes.numel():
            quality_target, _ = _box_iou_and_giou(
                boxes[image_index, :, None, :], target_boxes[None, :, :]
            )
            quality_target = quality_target.max(dim=1).values
        else:
            quality_target = torch.zeros(
                query_count, device=device, dtype=boxes.dtype)
        top_count = min(options.iqcm_top_m, query_count)
        top_indices = probabilities.max(dim=-1).values.topk(top_count).indices
        all_quality_predictions.append(
            quality_logits[image_index, top_indices])
        all_quality_targets.append(quality_target[top_indices].detach())

        if source_indices.numel() == 0:
            continue

        predicted_boxes = boxes[image_index, source_indices]
        expected_boxes = target_boxes[target_indices]
        all_predicted.append(predicted_boxes)
        all_expected.append(expected_boxes)

        _, giou = _box_iou_and_giou(predicted_boxes, expected_boxes)
        giou_losses.append(
            (1 - giou).clamp_min(0).pow(options.hifrm_gamma).sum())

        areas = expected_boxes[:, 2:].prod(dim=-1)
        all_areas.append(areas)
        all_uncertainties.append(
            predicted_log_variance[image_index, source_indices])

        if decoder_states.shape[0] > 1:
            normalized_layers = F.normalize(
                decoder_states[:, image_index], dim=-1, eps=1e-8
            )
            final_states = normalized_layers[-1]
            labels = target_labels[target_indices]
            unmatched = (~matched_mask[image_index]).nonzero(
                as_tuple=False).flatten()
            for local_index, query_index in enumerate(source_indices.tolist()):
                positive_vectors = [normalized_layers[:-1, query_index]]
                same_class = source_indices[labels == labels[local_index]]
                same_class = same_class[same_class != query_index]
                if same_class.numel():
                    positive_vectors.append(final_states[same_class])
                positive_similarities = torch.cat(positive_vectors, dim=0) @ final_states[
                    query_index
                ]
                if unmatched.numel():
                    similarities = final_states[unmatched] @ final_states[query_index]
                    ambiguity = _ambiguity(probabilities[unmatched])
                    negative_rank = similarities + 0.25 * ambiguity
                    selected = negative_rank.topk(
                        min(options.hard_negative_k, int(unmatched.numel()))
                    ).indices
                    negative_similarities = similarities[selected]
                    temperature = 0.15 * (0.5 + gate[image_index].detach())
                    positive_logits = positive_similarities / temperature
                    negative_logits = negative_similarities / temperature
                    contrastive_losses.append(
                        -torch.logsumexp(positive_logits, dim=0)
                        + torch.logsumexp(
                            torch.cat((positive_logits, negative_logits)), dim=0
                        )
                    )

    zeros = boxes.sum() * 0.0
    if all_predicted:
        predicted = torch.cat(all_predicted)
        expected = torch.cat(all_expected)
        areas = torch.cat(all_areas)
        area_threshold = (3.0 * areas.median()).clamp(0.02, 0.15)
        small_mask = (areas < area_threshold).to(boxes.dtype)
        log_variance = torch.cat(all_uncertainties)
        centered = log_variance - log_variance.mean(dim=0, keepdim=True)
        precision = torch.exp(-centered.clamp(-1.5, 1.1))
        precision = precision / precision.mean().clamp_min(1e-8)
        absolute_error = (predicted - expected).abs()
        weighted_error = (
            small_mask[:, None] * precision * absolute_error
            + (1 - small_mask[:, None]) * absolute_error
        )
        per_box_l1 = (
            lue_ramp * weighted_error.mean(dim=-1)
            + (1 - lue_ramp) * absolute_error.mean(dim=-1)
        )
        losses = {
            "loss_bbox": per_box_l1.sum() / normalized_box_count,
            "loss_giou": torch.stack(giou_losses).sum() / normalized_box_count,
        }
        calibration_target = -1.5 * torch.exp(
            -torch.linalg.vector_norm(
                predicted[:, :2] - expected[:, :2], dim=-1
            )
            / (expected[:, 2:].amax(dim=-1).clamp_min(1e-8) / 2)
        ) - 4.2
        scu_values = F.mse_loss(
            log_variance.mean(dim=-1), calibration_target.detach(), reduction="none"
        )
        loss_scu = (scu_values * small_mask).sum() / normalized_box_count
        error_target = (
            absolute_error.detach().clamp_min(1e-8).log().clamp(-7.0, 0.0)
        )
        uncertainty_error = F.smooth_l1_loss(
            log_variance, error_target, reduction="none", beta=0.5
        ).mean(dim=-1)
        loss_lue = (uncertainty_error * small_mask).sum() / \
            normalized_box_count
        losses["loss_scu"] = loss_scu * lue_ramp
        losses["loss_lue"] = loss_lue * lue_ramp
    else:
        losses = {"loss_bbox": zeros, "loss_giou": zeros,
                  "loss_scu": zeros, "loss_lue": zeros}

    iqcm = (
        F.binary_cross_entropy_with_logits(
            torch.cat(all_quality_predictions),
            torch.cat(all_quality_targets).to(dtype=quality_logits.dtype),
        )
        if all_quality_predictions
        else zeros
    )
    losses["loss_iqcm"] = iqcm
    losses["loss_tversky"] = torch.stack(
        tversky_losses).mean() if tversky_losses else zeros
    losses["loss_dqcd"] = (
        torch.stack(contrastive_losses).mean() * dqcd_ramp
        if contrastive_losses
        else zeros
    )
    return losses


def _attach_model_improvements(model: nn.Module) -> None:
    transformer = getattr(model, "transformer", None)
    num_queries = getattr(model, "num_queries", None)
    hidden_dim = getattr(transformer, "d_model", None)
    if not isinstance(num_queries, int) or not isinstance(hidden_dim, int):
        raise RuntimeError(
            "RF-DETR compatibility error: model must expose integer num_queries and transformer.d_model."
        )
    try:
        transformer_signature = inspect.signature(transformer.forward)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "RF-DETR compatibility error: could not inspect transformer.forward."
        ) from exc
    query_feat_weight_supported = "query_feat_weight" in transformer_signature.parameters

    heads = ImprovementHeads(hidden_dim)
    model.add_module("improvement_heads", heads)
    transformer.register_parameter(
        "improvement_query_scale", nn.Parameter(torch.ones(num_queries))
    )
    original_forward = getattr(model.forward, "__func__", model.forward)

    def transformer_output_hook(
        module: nn.Module, _inputs: tuple[Any, ...], result: Any
    ) -> Any:
        if not _inputs:
            raise RuntimeError(
                "RF-DETR transformer was called without positional inputs.")
        if not isinstance(result, (tuple, list)) or not result:
            raise RuntimeError(
                "RF-DETR transformer returned an unsupported output structure.")
        states = result[0]
        if not isinstance(states, torch.Tensor) or states.ndim != 4:
            raise RuntimeError(
                "RF-DETR transformer output must expose decoder states [layers, batch, queries, dim]."
            )
        if not query_feat_weight_supported:
            query_count = states.shape[-2]
            if query_count % num_queries:
                raise RuntimeError(
                    f"RF-DETR query count {query_count} is not divisible by base count {num_queries}."
                )
            repeats = query_count // num_queries
            query_scale = getattr(module, "improvement_query_scale", None)
            if not isinstance(query_scale, torch.Tensor):
                raise RuntimeError(
                    "RF-DETR QAR query-scale parameter is unavailable.")
            scales = query_scale.clamp(0.5, 2.0).repeat(repeats)
            shape = (1,) * (states.ndim - 2) + (query_count, 1)
            states = states * scales.reshape(shape)
            result_items = list(result)
            result_items[0] = states
            result = tuple(result_items) if isinstance(
                result, tuple) else result_items
        module._improve_decoder_states = states
        return result

    transformer.register_forward_hook(transformer_output_hook)

    def query_scale_pre_hook(
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        argument_names = tuple(transformer_signature.parameters)
        query_name = "query_feat_weight"
        if query_name in kwargs:
            query = kwargs[query_name]
            location = "kwargs"
        else:
            query_position = argument_names.index(query_name)
            query = args[query_position] if query_position < len(
                args) else None
            location = "args"
        if not isinstance(query, torch.Tensor) or query.shape[-1] != hidden_dim:
            raise RuntimeError(
                "RF-DETR compatibility error: query_feat_weight is not a compatible tensor."
            )
        query_count = query.shape[-2]
        if query_count % num_queries:
            raise RuntimeError(
                f"RF-DETR query count {query_count} is not divisible by base count {num_queries}."
            )
        repeats = query_count // num_queries
        query_scale = getattr(module, "improvement_query_scale", None)
        if not isinstance(query_scale, torch.Tensor):
            raise RuntimeError(
                "RF-DETR QAR query-scale parameter is unavailable.")
        scales = query_scale.clamp(0.5, 2.0).repeat(repeats)
        shape = (1,) * (query.ndim - 2) + (query_count, 1)
        scaled_query = query * scales.reshape(shape)
        if location == "kwargs":
            kwargs[query_name] = scaled_query
        else:
            mutable_args = list(args)
            mutable_args[query_position] = scaled_query
            args = tuple(mutable_args)
        return args, kwargs

    if query_feat_weight_supported:
        transformer.register_forward_pre_hook(
            query_scale_pre_hook, with_kwargs=True)

    def improved_forward(
        _model: nn.Module, samples: Any, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        frequency_gate = _frequency_gate(samples)
        result = original_forward(_model, samples, *args, **kwargs)
        states = getattr(_model.transformer, "_improve_decoder_states", None)
        _model.transformer._improve_decoder_states = None
        if not isinstance(result, dict) or not isinstance(states, torch.Tensor):
            raise RuntimeError(
                "RF-DETR forward did not return the expected output dictionary.")
        last_layer = states[-1]
        if result["pred_logits"].shape[:2] != last_layer.shape[:2]:
            raise RuntimeError(
                "RF-DETR decoder query count does not match prediction query count."
            )
        heads = _model.improvement_heads
        quality_logits = heads.quality_head(last_layer).squeeze(-1)
        quality = quality_logits.sigmoid().clamp(1e-6, 1 - 1e-6)
        result["pred_quality"] = quality
        result["pred_quality_logits"] = quality_logits
        result["pred_log_variance"] = heads.uncertainty_head(last_layer)
        result["improve_decoder_states"] = states
        result["improve_frequency_gate"] = frequency_gate
        if not _model.training:
            calibrated = result["pred_logits"].sigmoid() * \
                quality.unsqueeze(-1)
            result["pred_logits"] = torch.logit(
                calibrated.clamp(1e-6, 1 - 1e-6))
        return result

    model.forward = MethodType(improved_forward, model)


def _version_checked_install(
    options: ImprovementConfig, weights: dict[str, float]
) -> None:
    try:
        version = importlib.metadata.version("rfdetr")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "RF-DETR is not installed in this Python environment.") from exc
    if version != SUPPORTED_RFDETR_VERSION:
        raise RuntimeError(
            f"Improved training is tested against rfdetr=={SUPPORTED_RFDETR_VERSION}; "
            f"found {version}. Install the pinned version before training."
        )
    _attach_builder_hooks(options, weights)


def _attach_builder_hooks(
    options: ImprovementConfig, weights: dict[str, float]
) -> None:
    module_model = importlib.import_module("rfdetr.training.module_model")
    if getattr(module_model, "_improvement_builders_installed", False):
        state = module_model._improvement_state
        state["options"] = options
        state["weights"] = weights
        return
    original_builder = module_model.build_model_from_config
    original_criterion_builder = module_model.build_criterion_from_config
    original_training_step = module_model.RFDETRModelModule.training_step
    state: dict[str, Any] = {"options": options, "weights": weights}

    def model_builder(model_config: Any, train_config: Any) -> nn.Module:
        built_model = original_builder(model_config, train_config)
        _attach_model_improvements(built_model)
        return built_model

    def criterion_builder(model_config: Any, train_config: Any) -> tuple[Any, Any]:
        criterion, postprocess = original_criterion_builder(
            model_config, train_config)
        weight_dict = getattr(criterion, "weight_dict", None)
        if not isinstance(weight_dict, dict):
            raise RuntimeError(
                "RF-DETR compatibility error: criterion.weight_dict is unavailable."
            )
        weight_dict.update(state["weights"])
        old_forward = criterion.forward

        def improved_criterion_forward(
            self: nn.Module,
            outputs: dict[str, Any],
            targets: list[dict[str, torch.Tensor]],
            *criterion_args: Any,
            **criterion_kwargs: Any,
        ) -> dict[str, torch.Tensor]:
            base_losses = old_forward(
                outputs, targets, *criterion_args, **criterion_kwargs
            )
            active_options: ImprovementConfig = state["options"]
            extra_losses = _custom_losses(
                self,
                active_options,
                outputs,
                targets,
            )
            return _merge_loss_dicts(
                base_losses,
                extra_losses,
                weight_dict=self.weight_dict,
            )

        criterion.forward = MethodType(improved_criterion_forward, criterion)
        return criterion, postprocess

    module_model.build_model_from_config = model_builder
    module_model.build_criterion_from_config = criterion_builder
    module_model._improvement_builders_installed = True
    module_model._improvement_state = state

    def update_improvement_epoch(
        lightning_module: nn.Module, batch: Any, batch_index: int
    ) -> Any:
        criterion = getattr(lightning_module, "criterion", None)
        if criterion is not None:
            criterion._improve_epoch = int(lightning_module.current_epoch)
        return original_training_step(lightning_module, batch, batch_index)

    module_model.RFDETRModelModule.training_step = update_improvement_epoch


def _merge_loss_dicts(
    base_losses: dict[str, torch.Tensor],
    extra_losses: dict[str, torch.Tensor],
    *,
    weight_dict: dict[str, float],
) -> dict[str, torch.Tensor]:
    """Validate the RF-DETR loss contract before combining stock and custom losses.

    The custom LUE/HIFRM implementations intentionally replace only the final-layer
    ``loss_bbox`` and ``loss_giou`` values. Stock auxiliary losses such as
    ``loss_bbox_0`` and ``loss_giou_0`` remain untouched in ``base_losses``.
    RF-DETR silently ignores loss values without a matching ``weight_dict`` entry,
    so fail early instead of training with a partially inactive objective.
    """
    required_base = {"loss_ce", "loss_bbox", "loss_giou"}
    missing_base = required_base.difference(base_losses)
    missing_base_weights = required_base.difference(weight_dict)
    if missing_base:
        raise RuntimeError(
            "RF-DETR criterion compatibility error: missing base losses "
            f"{sorted(missing_base)}."
        )
    if missing_base_weights:
        raise RuntimeError(
            "RF-DETR criterion compatibility error: missing weights for base losses "
            f"{sorted(missing_base_weights)}."
        )

    missing_extra_weights = set(extra_losses).difference(weight_dict)
    if missing_extra_weights:
        raise RuntimeError(
            "RF-DETR improvement losses have no criterion weights: "
            f"{sorted(missing_extra_weights)}."
        )

    # Updating only exact keys means auxiliary keys with suffixes (_0, _1, _enc)
    # are retained. Guard that invariant explicitly because RF-DETR sums every
    # weighted auxiliary loss independently during training.
    auxiliary_before = {
        key: value
        for key, value in base_losses.items()
        if key.startswith(("loss_bbox_", "loss_giou_"))
    }
    missing_auxiliary_weights = set(auxiliary_before).difference(weight_dict)
    if missing_auxiliary_weights:
        raise RuntimeError(
            "RF-DETR returned auxiliary losses without criterion weights: "
            f"{sorted(missing_auxiliary_weights)}."
        )
    merged = dict(base_losses)
    merged.update(extra_losses)
    for key, value in auxiliary_before.items():
        if merged.get(key) is not value:
            raise RuntimeError(
                f"RF-DETR auxiliary loss {key!r} was unexpectedly overwritten."
            )
    return merged


def _improvement_config(raw: dict[str, Any]) -> tuple[ImprovementConfig, dict[str, float]]:
    improvement_raw = raw.get("improvements", {})
    options = ImprovementConfig(
        dqcd_weight=float(improvement_raw.get("dqcd_weight", 0.08)),
        iqcm_weight=float(improvement_raw.get("iqcm_weight", 0.25)),
        scu_weight=float(improvement_raw.get("scu_weight", 0.1)),
        lue_weight=float(improvement_raw.get("lue_weight", 0.25)),
        tversky_weight=float(improvement_raw.get("tversky_weight", 0.2)),
        tversky_alpha=float(improvement_raw.get("tversky_alpha", 0.7)),
        tversky_beta=float(improvement_raw.get("tversky_beta", 0.3)),
        hifrm_gamma=float(improvement_raw.get("hifrm_gamma", 2.0)),
        hard_negative_k=int(improvement_raw.get("hard_negative_k", 128)),
        iqcm_top_m=int(improvement_raw.get("iqcm_top_m", 100)),
        dqcd_start_epoch=int(improvement_raw.get("dqcd_start_epoch", 8)),
        dqcd_warmup_epochs=int(improvement_raw.get("dqcd_warmup_epochs", 5)),
        lue_warmup_epochs=int(improvement_raw.get("lue_warmup_epochs", 15)),
    )
    if (
        options.hard_negative_k < 1
        or options.iqcm_top_m < 1
        or options.dqcd_warmup_epochs < 1
        or options.lue_warmup_epochs < 1
        or options.dqcd_start_epoch < 0
        or any(
            not math.isfinite(value) or value <= 0
            for value in (
                options.hifrm_gamma,
                options.tversky_alpha,
                options.tversky_beta,
            )
        )
    ):
        raise ValueError(
            "Improvement loss counts, warmups, and gamma must be positive.")
    configured_losses = raw.get("loss_weights", {})
    if not isinstance(configured_losses, dict):
        raise ValueError(
            "loss_weights must be a mapping in the improvement YAML.")
    if "loss_vfl" in configured_losses and "loss_ce" in configured_losses:
        if float(configured_losses["loss_vfl"]) != float(configured_losses["loss_ce"]):
            raise ValueError(
                "Specify only loss_weights.loss_ce; RF-DETR 1.11.2 names its "
                "classification loss 'loss_ce', not 'loss_vfl'."
            )
    if "loss_vfl" in configured_losses and "loss_ce" not in configured_losses:
        # Backward compatibility for older project YAMLs; this is an alias only.
        print(
            "[WARNING] loss_weights.loss_vfl is a legacy config name; "
            "mapping it to RF-DETR's criterion key 'loss_ce'."
        )
    weights = {
        "loss_ce": float(
            configured_losses.get(
                "loss_ce", configured_losses.get("loss_vfl", 1.0))
        ),
        "loss_bbox": float(configured_losses.get("loss_bbox", 5.0)),
        "loss_giou": float(configured_losses.get("loss_giou", 2.0)),
        "loss_dqcd": options.dqcd_weight,
        "loss_iqcm": options.iqcm_weight,
        "loss_scu": options.scu_weight,
        "loss_lue": options.lue_weight,
        "loss_tversky": options.tversky_weight,
    }
    if any(not math.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError(
            "Improvement loss weights must be finite and non-negative.")
    return options, weights


def _read_validation_map(output_dir: Path) -> float:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_files = sorted(output_dir.rglob("events.out.tfevents.*"))
    candidates: list[tuple[int, float]] = []
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file))
        accumulator.Reload()
        tags = accumulator.Tags().get("scalars", [])
        tag = next(
            (candidate for candidate in ("val/mAP_50_95",
             "val/mAP50_95") if candidate in tags),
            None,
        )
        if tag:
            candidates.extend(
                (event.step, float(event.value)) for event in accumulator.Scalars(tag)
            )
    if not candidates:
        raise RuntimeError(
            f"No validation mAP@0.5:0.95 TensorBoard scalar found under {output_dir}."
        )
    score = max(candidates, key=lambda item: item[0])[1]
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise RuntimeError(
            f"Validation mAP is outside the expected [0, 1] range: {score}.")
    return score


def _rfo_initial_population(
    settings: dict[str, Any], rng: random.Random
) -> list[dict[str, float | int]]:
    population_size = int(settings.get("population", 3))
    if population_size < 2:
        raise ValueError("RFO population must be >= 2.")
    bounds = {
        "lr": (math.log(1e-5), math.log(3e-4)),
        "weight_decay": (math.log(1e-6), math.log(1e-3)),
    }
    query_choices = [100, 200, 300]
    batch_choices = [4, 8, 16]

    def random_fox() -> dict[str, float | int]:
        return {
            "lr": math.exp(rng.uniform(*bounds["lr"])),
            "weight_decay": math.exp(rng.uniform(*bounds["weight_decay"])),
            "num_queries": rng.choice(query_choices),
            "batch_size": rng.choice(batch_choices),
        }

    return [random_fox() for _ in range(population_size)]


def _rfo_next_generation(
    population: list[dict[str, float | int]],
    best: dict[str, float | int],
    rng: random.Random,
) -> list[dict[str, float | int]]:
    """Exploit around the best validation candidate while retaining bounded exploration."""
    bounds = {
        "lr": (math.log(1e-5), math.log(3e-4)),
        "weight_decay": (math.log(1e-6), math.log(1e-3)),
    }
    query_choices = [100, 200, 300]
    batch_choices = [4, 8, 16]
    next_population: list[dict[str, float | int]] = []
    for fox in population:
        next_fox: dict[str, float | int] = {}
        for key, (lower, upper) in bounds.items():
            current = math.log(float(fox[key]))
            prey = math.log(float(best[key]))
            distance = abs(prey - current)
            step = rng.random() * distance
            exploration = (rng.random() - 0.5) * (upper - lower) * 0.2
            next_fox[key] = math.exp(
                min(upper, max(lower, current + rng.random() * step + exploration))
            )
        next_fox["num_queries"] = rng.choice(
            [int(best["num_queries"]), rng.choice(query_choices)]
        )
        next_fox["batch_size"] = rng.choice(
            [int(best["batch_size"]), rng.choice(batch_choices)]
        )
        next_population.append(next_fox)
    return next_population


def _model_class(variant: str) -> type[Any]:
    package = importlib.import_module("rfdetr")
    model_class = getattr(package, variant, None)
    if not isinstance(model_class, type):
        raise ValueError(f"Unsupported RF-DETR model variant: {variant!r}.")
    return model_class


def _make_model(
    model_class: type[Any],
    resolution: int,
    num_queries: int,
    pretrained: bool,
) -> Any:
    kwargs: dict[str, Any] = {
        "resolution": resolution,
        "num_queries": num_queries,
        "compile": False,
        "cuda_graphs": False,
    }
    if not pretrained:
        kwargs["pretrain_weights"] = None
    return model_class(**kwargs)


def _training_kwargs(
    dataset_dir: str,
    training: dict[str, Any],
    output_dir: Path,
    candidate: dict[str, float | int],
    *,
    epochs: int,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    configured_device = training.get("device", "cuda")
    device = "cuda" if configured_device == 0 else configured_device
    effective_batch_size = int(candidate["batch_size"])
    physical_batch_limit = int(training.get("max_physical_batch_size", 4))
    if effective_batch_size < 1 or physical_batch_limit < 1:
        raise ValueError(
            "Batch sizes and max_physical_batch_size must be positive.")
    physical_batch_size = min(effective_batch_size, physical_batch_limit)
    accumulation = max(
        int(training.get("grad_accum_steps", 1)),
        math.ceil(effective_batch_size / physical_batch_size),
    )
    kwargs: dict[str, Any] = {
        "dataset_dir": dataset_dir,
        "epochs": epochs,
        "early_stopping": True,
        "early_stopping_patience": int(training.get("patience", 10)),
        "batch_size": physical_batch_size,
        "grad_accum_steps": accumulation,
        "lr": float(candidate["lr"]),
        "lr_encoder": float(training.get("lr_encoder", 1.5e-4)),
        "weight_decay": float(candidate["weight_decay"]),
        "output_dir": str(output_dir),
        "resolution": int(training.get("imgsz", 640)),
        "num_workers": int(training.get("workers", 4)),
        "device": device,
        "seed": int(training.get("seed", 42)),
        "eval_interval": 1,
        # Save a full resumable checkpoint after every epoch.
        "checkpoint_interval": 1,
        "tensorboard": True,
    }
    if resume_checkpoint is not None and resume_checkpoint.is_file():
        kwargs["resume"] = str(resume_checkpoint)
        print(f"[RESUME] Continuing from full checkpoint: {resume_checkpoint}")
    return kwargs


def _sync_checkpoint(output_dir: Path) -> None:
    candidates = (
        "checkpoint_best_ema.pth",
        "checkpoint_best_regular.pth",
        "checkpoint_best.pth",
        "best.pth",
    )
    for name in candidates:
        source = output_dir / name
        if source.is_file() and source.name != "best.pth":
            shutil.copy2(source, output_dir / "best.pth")
            break
    current = output_dir / "checkpoint.pth"
    if current.is_file():
        shutil.copy2(current, output_dir / "last.pth")


def _export_tensorboard_logs(
    output_dir: Path, epoch_logger: EpochLogger
) -> None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    values_by_step: dict[int, dict[str, float]] = {}
    event_files = (
        path
        for path in output_dir.rglob("events.out.tfevents.*")
        if "rfo_trials" not in path.parts
    )
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file))
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars(tag):
                values_by_step.setdefault(event.step, {})[
                    tag] = float(event.value)
    validation_steps = sorted(
        step
        for step, values in values_by_step.items()
        if "val/mAP_50_95" in values or "val/mAP50_95" in values
    )
    for epoch, step in enumerate(validation_steps, start=1):
        values = values_by_step[step]
        epoch_logger.logs_dict[epoch] = {
            "Epoch": epoch,
            "Train loss": round(values.get("train/loss", 0.0), 5),
            "Class loss": round(values.get("train/loss_ce", 0.0), 5),
            "Box loss": round(values.get("train/loss_bbox", 0.0), 5),
            "GIoU": round(values.get("train/loss_giou", 0.0), 5),
            "DQCD": round(values.get("train/loss_dqcd", 0.0), 5),
            "IQCM": round(values.get("train/loss_iqcm", 0.0), 5),
            "SCU": round(values.get("train/loss_scu", 0.0), 5),
            "LUE": round(values.get("train/loss_lue", 0.0), 5),
            "Tversky": round(values.get("train/loss_tversky", 0.0), 5),
            "Learning rate": f"{values.get('train/lr', 1e-4):.6e}",
            "val_mAP50": round(values.get("val/mAP_50", 0.0), 4),
            "val_mAP50_95": round(
                values.get("val/mAP_50_95", values.get("val/mAP50_95", 0.0)), 4
            ),
        }
    epoch_logger.save_csv()


def _validate_dataset(dataset_dir: str) -> str:
    path = Path(dataset_dir)
    has_coco = all(
        (path / split / "_annotations.coco.json").is_file()
        for split in ("train", "valid")
    )
    has_yolo = (
        (path / "data.yaml").is_file()
        and (path / "train" / "images").is_dir()
    )
    if not has_coco and not has_yolo:
        raise FileNotFoundError(
            f"Expected COCO train/valid annotations or YOLO data.yaml/images under {path}."
        )
    return str(path)


def train_rfdetr(
    dataset_dir: str | None = None,
    data_dir: str | None = None,
    config_path: str | Path | None = None,
    output_dir: str | None = None,
    log_dir: str | None = None,
    commit_fn: Callable[[], None] | None = None,
) -> str:
    config_file = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not config_file.is_file():
        raise FileNotFoundError(
            f"RF-DETR improvement config not found: {config_file}")
    with config_file.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    model_config = config.get("model", {})
    training = config.get("training", {})
    output_config = config.get("outputs", {})
    if training.get("use_balanced_loss", False):
        print(
            "[WARNING] use_balanced_loss is not a supported RF-DETR argument; "
            "the integrated Tversky loss handles the requested class-balance term."
        )
    dataset_dir = _validate_dataset(
        dataset_dir or data_dir or "/vol/datasets/completed-project-5/coco"
    )
    checkpoint_dir = Path(
        output_dir or output_config.get(
            "checkpoint_dir", "/vol/checkpoints/rfdetr_improve_v1")
    )
    logs_dir = Path(log_dir or output_config.get(
        "log_dir", "/vol/logs/rfdetr_improve_v1"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    variant = str(model_config.get("variant", "RFDETRMedium"))
    resolution = int(training.get("imgsz", 640))
    pretrained = bool(model_config.get("pretrained", True))
    seed = int(training.get("seed", 42))
    options, weights = _improvement_config(config)
    _version_checked_install(options, weights)
    model_class = _model_class(variant)

    original_sigterm = signal.getsignal(signal.SIGTERM)
    epoch_logger = EpochLogger(
        log_file_path=str(logs_dir / "rfdetr_improve_epoch_logs.csv"),
        model_name="RF-DETR-Improved",
    )

    def sync_final_outputs() -> None:
        _sync_checkpoint(checkpoint_dir)
        _export_tensorboard_logs(checkpoint_dir, epoch_logger)
        if commit_fn is not None:
            commit_fn()

    def handle_sigterm(signum: int, _: Any) -> None:
        raise SystemExit(128 + signum)

    # Modal Volumes persist only after commit(). RF-DETR writes checkpoint.pth
    # during training; commit it once the file size/mtime has stopped changing.
    commit_stop = threading.Event()

    def checkpoint_commit_worker() -> None:
        checkpoint_file = checkpoint_dir / "checkpoint.pth"
        last_committed: tuple[int, int] | None = None
        previous: tuple[int, int] | None = None
        stable_observations = 0
        while not commit_stop.wait(20):
            try:
                stat = checkpoint_file.stat()
                signature = (stat.st_size, stat.st_mtime_ns)
            except OSError:
                previous = None
                stable_observations = 0
                continue
            if signature == previous:
                stable_observations += 1
            else:
                previous = signature
                stable_observations = 0
            if (stable_observations >= 1 and signature != last_committed
                    and signature[0] > 0 and commit_fn is not None):
                try:
                    commit_fn()
                    last_committed = signature
                    print(
                        f"[CHECKPOINT] Modal Volume committed: {checkpoint_file} ({signature[0]} bytes)")
                except Exception as exc:
                    # Keep training alive; the next stable checkpoint will retry.
                    print(f"[WARNING] Periodic volume commit failed: {exc!r}")

    commit_thread = threading.Thread(
        target=checkpoint_commit_worker,
        name="rfdetr-checkpoint-committer",
        daemon=True,
    )
    if commit_fn is not None:
        commit_thread.start()

    signal.signal(signal.SIGTERM, handle_sigterm)
    try:
        rfo = dict(config.get("rfo", {}))
        if bool(rfo.get("enabled", True)):
            trial_epochs = int(rfo.get("epochs_per_trial", 3))
            generations = int(rfo.get("generations", 2))
            if trial_epochs < 1 or generations < 1:
                raise ValueError("RFO epochs_per_trial must be >= 1.")
            rng = random.Random(seed)
            population = _rfo_initial_population(rfo, rng)
            population[0] = {
                "lr": float(training.get("lr", 1e-4)),
                "weight_decay": float(training.get("weight_decay", 1e-4)),
                "num_queries": int(model_config.get("num_queries", 300)),
                "batch_size": int(training.get("batch_size", 16)),
            }
            results: list[dict[str, Any]] = []
            best_candidate: dict[str, float | int] | None = None
            best_score = -math.inf
            trials_root = (
                checkpoint_dir / "rfo_trials" / f"run_{uuid.uuid4().hex[:8]}"
            )
            trials_root.mkdir(parents=True, exist_ok=True)
            print(
                f"[*] RFO validation search: {len(population) * generations} candidates, "
                f"{trial_epochs} epochs each; objective=mAP@0.5:0.95."
            )
            index = 0
            for generation in range(generations):
                for candidate in population:
                    index += 1
                    trial_dir = trials_root / f"candidate_{index:03d}"
                    trial_dir.mkdir(parents=True, exist_ok=False)
                    model = _make_model(
                        model_class,
                        resolution,
                        int(candidate["num_queries"]),
                        pretrained,
                    )
                    model.train(
                        **_training_kwargs(
                            dataset_dir,
                            training,
                            trial_dir,
                            candidate,
                            epochs=trial_epochs,
                        )
                    )
                    score = _read_validation_map(trial_dir)
                    result = {
                        "generation": generation + 1,
                        "candidate": index,
                        **candidate,
                        "val_mAP50_95": score,
                    }
                    results.append(result)
                    if score > best_score:
                        best_score = score
                        best_candidate = dict(candidate)
                    print(f"[RFO] {result}")
                    shutil.rmtree(trial_dir)
                    with (logs_dir / "rfo_results.csv").open(
                        "w", newline="", encoding="utf-8"
                    ) as result_file:
                        writer = csv.DictWriter(
                            result_file, fieldnames=list(results[0])
                        )
                        writer.writeheader()
                        writer.writerows(results)
                    if commit_fn is not None:
                        commit_fn()
                if generation + 1 < generations:
                    if best_candidate is None:
                        raise RuntimeError(
                            "RFO has no validation-scored candidate to follow."
                        )
                    population = _rfo_next_generation(
                        population, best_candidate, rng
                    )
            if best_candidate is None:
                raise RuntimeError(
                    "RFO completed without a validation-scored candidate.")
            (logs_dir / "rfo_best.json").write_text(
                json.dumps(
                    {"validation_map50_95": best_score, **best_candidate},
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            best_candidate = {
                "lr": float(training.get("lr", 1e-4)),
                "weight_decay": float(training.get("weight_decay", 1e-4)),
                "num_queries": int(model_config.get("num_queries", 300)),
                "batch_size": int(training.get("batch_size", 4)),
            }

        final_model = _make_model(
            model_class,
            resolution,
            int(best_candidate["num_queries"]),
            pretrained,
        )
        print(
            f"[*] Final improved training: {variant}, queries={best_candidate['num_queries']}, "
            f"batch={best_candidate['batch_size']}, lr={best_candidate['lr']:.3g}."
        )
        resume_enabled = bool(training.get("resume", True))
        resume_checkpoint = checkpoint_dir / "checkpoint.pth"
        if not resume_enabled:
            resume_checkpoint = None
        elif not resume_checkpoint.is_file():
            print(
                "[RESUME] No existing full checkpoint found; starting a new final training run.")
            resume_checkpoint = None

        final_model.train(
            **_training_kwargs(
                dataset_dir,
                training,
                checkpoint_dir,
                best_candidate,
                epochs=int(training.get("epochs", 80)),
                resume_checkpoint=resume_checkpoint,
            )
        )
    finally:
        commit_stop.set()
        if commit_thread.is_alive():
            commit_thread.join(timeout=25)
        signal.signal(signal.SIGTERM, original_sigterm)
        sync_final_outputs()

    best_checkpoint = checkpoint_dir / "best.pth"
    if not best_checkpoint.is_file():
        raise FileNotFoundError(
            f"RF-DETR training did not produce a best checkpoint at {best_checkpoint}."
        )
    return str(best_checkpoint)


if __name__ == "__main__":
    train_rfdetr()
