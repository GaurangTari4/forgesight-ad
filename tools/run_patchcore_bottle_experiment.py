"""Real-data, normal-only PatchCore bottle experiment (ResNet-18)."""
from __future__ import annotations
import json,sys,time,hashlib
from pathlib import Path
import torch
from torch.utils.data import DataLoader,Subset
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));DATA=ROOT.parent/'mvtec-ad'/'versions'/'2';SPLIT=ROOT/'artifacts'/'v2_experiments'/'resnet50_bottle_v2_seed42'/'split_manifest.json';OUT=ROOT/'artifacts'/'patchcore_experiments'/'resnet18_bottle_seed42'
from forgesight_ad.autoencoder_pipeline import MVTecTrainDataset
from forgesight_ad.backbone_pipeline import build_resnet_extractor,build_transforms
from forgesight_ad.patchcore_pipeline import METHODOLOGY_ID,patch_embeddings,greedy_coreset,save_artifact,load_artifact

def filehash(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 if OUT.exists():raise FileExistsError(OUT)
 d=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');manifest=json.loads(SPLIT.read_text());fit=set(manifest['fit_indices']);cal=set(manifest['calibration_indices']);assert fit.isdisjoint(cal) and len(fit)==167 and len(cal)==42
 transform,_=build_transforms(512);train=MVTecTrainDataset(DATA,'bottle',transform);model,source=build_resnet_extractor('resnet18',d,False);layers=('layer2','layer3');loader=DataLoader(Subset(train,sorted(fit)),batch_size=2,shuffle=False,pin_memory=True);torch.cuda.reset_peak_memory_stats(d);start=time.perf_counter();chunks=[]
 with torch.inference_mode():
  for images,_ in loader:
   p,_=patch_embeddings(model(images.to(d)),layers);chunks.append(p.cpu())
 candidates=torch.cat(chunks);candidate_count=min(8192,len(candidates));candidate=candidates[torch.linspace(0,len(candidates)-1,candidate_count).long()].to(d);idx=greedy_coreset(candidate,256,42);bank=candidate[idx].cpu().contiguous();fit_seconds=time.perf_counter()-start
 payload={'methodology_id':METHODOLOGY_ID,'backbone_name':'resnet18','backbone_source':source,'feature_layers':list(layers),'image_size':512,'embedding_normalization':'l2','memory_bank':bank,'coreset':{'algorithm':'deterministic_greedy_kcenter_on_uniform_candidate_subset','candidate_count':candidate_count,'selected_count':len(bank)},'seed':42,'split_manifest_sha256':filehash(SPLIT),'calibration':{'status':'pending_normal_only','indices':sorted(cal)},'fit_images':len(fit),'total_patches':len(candidates),'fit_seconds':fit_seconds,'peak_cuda_memory_bytes':int(torch.cuda.max_memory_allocated(d))}
 save_artifact(OUT/'model.pt',payload);loaded=load_artifact(OUT/'model.pt',d,'resnet18');(OUT/'metadata.json').write_text(json.dumps({k:v for k,v in payload.items() if k!='memory_bank'},indent=2));print(json.dumps({'bank_shape':list(loaded['memory_bank'].shape),'total_patches':len(candidates),'fit_seconds':fit_seconds,'peak_cuda_memory_bytes':payload['peak_cuda_memory_bytes']},indent=2))
if __name__=='__main__':main()
