"""Safe standalone inference for versioned feature-statistics-v2 artifacts."""
from __future__ import annotations
from dataclasses import fields
from pathlib import Path
from typing import Any
import torch

from .autoencoder_pipeline import gaussian_blur, refine_predicted_mask, topk_mean_scores
from .backbone_pipeline import ResNetBackboneConfig, build_resnet_extractor, estimate_foreground_mask
from .feature_statistics_v2 import METHODOLOGY_ID, score


class V2ArtifactError(ValueError): pass


def load_v2_artifact(path: Path, device: torch.device, selected_backbone: str | None = None) -> tuple[dict[str, Any], ResNetBackboneConfig, torch.nn.Module]:
    try: payload = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc: raise V2ArtifactError(f"Unable to safely load v2 artifact: {path.name}") from exc
    if not isinstance(payload, dict) or payload.get("methodology_id") != METHODOLOGY_ID: raise V2ArtifactError("Artifact is not feature_statistics_v2_unweighted_layer_stats")
    if not isinstance(payload.get("statistics"), dict) or payload["statistics"].get("methodology_id") != METHODOLOGY_ID: raise V2ArtifactError("Artifact statistics lack the v2 methodology identifier")
    raw=payload.get("config"); required={"backbone_name","image_size","feature_layers","score_blend_weights","embedding_eps"}
    if not isinstance(raw,dict) or required.difference(raw): raise V2ArtifactError("Artifact configuration is incomplete")
    if selected_backbone and raw["backbone_name"] != selected_backbone: raise V2ArtifactError("Selected backbone does not match v2 artifact")
    allowed={item.name for item in fields(ResNetBackboneConfig)}; values={key:value for key,value in raw.items() if key in allowed}; values["device_name"]=str(device)
    for key in ("project_root", "dataset_root", "artifacts_root"):
        values[key] = Path(values[key])
    for key in ("feature_layers","feature_blend_weights","score_blend_weights"): values[key]=tuple(values[key])
    config=ResNetBackboneConfig(**values); extractor,_=build_resnet_extractor(config.backbone_name,device,allow_random_init=False); extractor.eval()
    return payload,config,extractor


@torch.inference_mode()
def infer_v2(payload: dict, config: ResNetBackboneConfig, extractor, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if images.device != config.device: raise V2ArtifactError("Input tensor and v2 extractor device differ")
    anomaly=score(extractor,images,payload["statistics"],config.feature_layers,config.score_blend_weights,config.embedding_eps)
    anomaly=gaussian_blur(anomaly,config.heatmap_blur_kernel,config.heatmap_blur_sigma)
    padded=(images.max(1,keepdim=True).values <= (0.5/255)).float()
    if torch.any(padded): anomaly=anomaly*(1-torch.nn.functional.max_pool2d(padded,15,1,7))
    if config.suppress_background_anomalies: anomaly=anomaly*estimate_foreground_mask(images,float(config.background_tolerance),float(config.background_min_coverage))
    margin=max(12,images.shape[-1]//12); anomaly[...,:margin,:]=0; anomaly[...,-margin:,:]=0; anomaly[...,:,:margin]=0; anomaly[...,:,-margin:]=0
    scores=topk_mean_scores(anomaly,config.top_k_ratio); mask=refine_predicted_mask(anomaly,float(payload["pixel_threshold"]),config.mask_smoothing_kernel,config.mask_min_area_ratio,config.max_mask_components)
    if not torch.isfinite(anomaly).all() or not torch.isfinite(scores).all(): raise V2ArtifactError("V2 inference produced non-finite values")
    return anomaly,scores,mask
