"""Compare the experiment v2 path with standalone v2 artifact inference."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from forgesight_ad.backbone_pipeline import build_transforms
from forgesight_ad.v2_inference import infer_v2,load_v2_artifact
from run_v2_bottle_experiment import maps

artifact=ROOT/'artifacts'/'v2_experiments'/'resnet50_bottle_v2_seed42'/'model.pt'; image=ROOT.parent/'mvtec-ad'/'versions'/'2'/'bottle'/'test'/'good'/'000.png'; out=ROOT/'validation_runs'/'v2_equivalence'
def main():
 out.mkdir(parents=True,exist_ok=True); device=torch.device('cuda:0'); payload,cfg,extractor=load_v2_artifact(artifact,device,'resnet50'); transform,_=build_transforms(cfg.image_size); tensor=transform(Image.open(image).convert('RGB')).unsqueeze(0).to(device)
 with torch.inference_mode(): legacy_map,legacy_score=maps(extractor,tensor,payload['statistics'],cfg); new_map,new_score,new_mask=infer_v2(payload,cfg,extractor,tensor)
 legacy_mask=(legacy_map>float(payload['pixel_threshold'])).float() # threshold map; standalone uses refined mask, compare scorer maps/scores only
 delta=(legacy_map-new_map).abs(); result={'artifact':str(artifact),'image':str(image),'device':str(device),'map_shape':list(new_map.shape),'score_shape':list(new_score.shape),'max_absolute_error':float(delta.max()),'max_relative_error':float((delta/(legacy_map.abs()+1e-12)).max()),'score_absolute_error':float((legacy_score-new_score).abs().max()),'threshold_equal':float(payload['image_threshold'])==float(payload['image_threshold']),'map_exact_equal':bool(torch.equal(legacy_map,new_map)),'score_exact_equal':bool(torch.equal(legacy_score,new_score)),'standalone_mask_shape':list(new_mask.shape),'finite':bool(torch.isfinite(new_map).all() and torch.isfinite(new_score).all())}
 (out/'equivalence.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
