"""Comparable warm batch-1 CUDA latency profile for v2 and PatchCore."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));OUT=ROOT/'validation_runs'/'warm_latency_20261030';OUT.mkdir(parents=True,exist_ok=False)
from forgesight_ad.backbone_pipeline import build_transforms,build_resnet_extractor
from forgesight_ad.v2_inference import load_v2_artifact,infer_v2
from forgesight_ad.patchcore_pipeline import load_artifact,anomaly_map
def stats(x):return {k:float(v) for k,v in {'mean':np.mean(x),'median':np.median(x),'p95':np.quantile(x,.95),'p99':np.quantile(x,.99),'min':np.min(x),'max':np.max(x)}.items()}
def measure(name,fn):
 for _ in range(10):fn();torch.cuda.synchronize()
 torch.cuda.reset_peak_memory_stats();times=[]
 for _ in range(50):
  a=torch.cuda.Event(True);b=torch.cuda.Event(True);a.record();fn();b.record();torch.cuda.synchronize();times.append(a.elapsed_time(b))
 return {'detector':name,'iterations':50,'gpu_full_inference_ms':stats(times),'peak_cuda_memory_bytes':int(torch.cuda.max_memory_allocated()),'samples_ms':times}
def main():
 d=torch.device('cuda:0');im=Image.open(ROOT.parent/'mvtec-ad/versions/2/bottle/test/good/000.png').convert('RGB');t,_=build_transforms(512);x=t(im).unsqueeze(0).to(d)
 va,vc,ve=load_v2_artifact(ROOT/'artifacts/v2_experiments/resnet50_bottle_v2_seed42/model.pt',d,'resnet50');pa=load_artifact(ROOT/'artifacts/patchcore_experiments/resnet18_bottle_seed42/model.pt',d,'resnet18');pe,_=build_resnet_extractor('resnet18',d,False)
 result={'protocol':'batch-1, preloaded artifacts/models, 10 warmups, CUDA-event full GPU detector path; excludes PIL decoding/preprocessing','v2':measure('v2',lambda:infer_v2(va,vc,ve,x)),'patchcore':measure('patchcore',lambda:anomaly_map(pe,x,pa['memory_bank'],tuple(pa['feature_layers'])))}; (OUT/'latency.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='protocol'},indent=2))
if __name__=='__main__':main()
