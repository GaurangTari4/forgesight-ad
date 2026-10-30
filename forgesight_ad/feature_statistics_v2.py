"""Versioned, mathematically consistent feature-statistics detector.

Unlike the legacy implementation, feature-layer weights are applied only when
combining per-layer anomaly maps; fitted means and standard deviations describe
the unweighted tensors that are compared at inference.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .autoencoder_pipeline import imagenet_normalize


METHODOLOGY_ID = "feature_statistics_v2_unweighted_layer_stats"


def _weights(weights: tuple[float, ...], count: int) -> list[float]:
    if len(weights) != count or sum(weights) <= 0:
        raise ValueError("Feature weights must be positive and match feature layers")
    total = float(sum(weights))
    return [float(item / total) for item in weights]


@torch.inference_mode()
def fit_statistics(extractor, images, feature_layers: tuple[str, ...], eps: float) -> dict:
    sums: dict[str, torch.Tensor] = {}; squares: dict[str, torch.Tensor] = {}; count = 0; target_size = None
    for batch in images:
        tensor = batch[0] if isinstance(batch, (tuple, list)) else batch
        parameter = next(iter(extractor.parameters()), None) if hasattr(extractor, "parameters") else None
        if parameter is not None:
            tensor = tensor.to(parameter.device, non_blocking=True)
        features = extractor(imagenet_normalize(tensor))
        target_size = target_size or tuple(features[feature_layers[0]].shape[-2:])
        for layer in feature_layers:
            fmap = features[layer]
            if fmap.shape[-2:] != target_size: fmap = F.interpolate(fmap, size=target_size, mode="bilinear", align_corners=False)
            sums[layer] = sums.get(layer, torch.zeros_like(fmap[0])) + fmap.sum(0)
            squares[layer] = squares.get(layer, torch.zeros_like(fmap[0])) + fmap.square().sum(0)
        count += tensor.shape[0]
    if count == 0: raise ValueError("Cannot fit feature statistics from an empty normal-only dataset")
    return {"methodology_id": METHODOLOGY_ID, "target_size": target_size, "layers": {layer: {"mean": (sums[layer] / count).cpu(), "std": torch.sqrt(torch.clamp(squares[layer] / count - (sums[layer] / count).square(), min=eps)).cpu()} for layer in feature_layers}}


@torch.inference_mode()
def score(extractor, images: torch.Tensor, statistics: dict, feature_layers: tuple[str, ...], blend_weights: tuple[float, ...], eps: float) -> torch.Tensor:
    if statistics.get("methodology_id") != METHODOLOGY_ID: raise ValueError("v2 scorer requires v2 statistics")
    features = extractor(imagenet_normalize(images)); target = tuple(statistics["target_size"]); maps = []
    for layer, weight in zip(feature_layers, _weights(blend_weights, len(feature_layers))):
        fmap = features[layer]
        if fmap.shape[-2:] != target: fmap = F.interpolate(fmap, size=target, mode="bilinear", align_corners=False)
        stat = statistics["layers"][layer]; z = (fmap - stat["mean"].to(images.device)) / (stat["std"].to(images.device) + eps)
        maps.append(weight * F.interpolate(z.square().mean(1, keepdim=True), size=images.shape[-2:], mode="bilinear", align_corners=False))
    return torch.stack(maps).sum(0)
