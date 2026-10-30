"""Read-only distribution audit for the prespecified v2 normal-only threshold."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader,Subset
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from forgesight_ad.autoencoder_pipeline import MVTecTestDataset,MVTecTrainDataset
from forgesight_ad.backbone_pipeline import build_transforms
from forgesight_ad.v2_inference import load_v2_artifact,infer_v2
artifact=ROOT/'artifacts'/'v2_experiments'/'resnet50_bottle_v2_seed42'; data=ROOT.parent/'mvtec-ad'/'versions'/'2'
def desc(x): return {'count':int(len(x)),'min':float(x.min()),'p50':float(np.quantile(x,.5)),'p95':float(np.quantile(x,.95)),'p98':float(np.quantile(x,.98)),'max':float(x.max())}
def main():
 p,c,e=load_v2_artifact(artifact/'model.pt',torch.device('cuda:0'),'resnet50');t,m=build_transforms(c.image_size);manifest=json.loads((artifact/'split_manifest.json').read_text());train=MVTecTrainDataset(data,'bottle',t);cal=DataLoader(Subset(train,manifest['calibration_indices']),batch_size=2);test=MVTecTestDataset(data,'bottle',t,m,c.image_size);testl=DataLoader(test,batch_size=2);
 def run(loader,labeled=False):
  ss=[];yy=[]
  with torch.inference_mode():
   for b in loader:
    x=(b['image'] if isinstance(b,dict) else b[0]).to(c.device);_,s,_=infer_v2(p,c,e,x);ss.extend(s.cpu().tolist());
    if labeled:yy.extend(b['label'].tolist())
  return np.asarray(ss),np.asarray(yy)
 cal_s,_=run(cal);test_s,y=run(testl,True);th=float(p['image_threshold']);out={'threshold':th,'calibration':desc(cal_s),'normal_test':desc(test_s[y==0]),'defective_test':desc(test_s[y==1]),'calibration_scores_at_or_above_threshold':int((cal_s>=th).sum()),'normal_test_at_or_above_threshold':int((test_s[y==0]>=th).sum()),'defective_test_below_threshold':int((test_s[y==1]<th).sum()),'note':'Test distributions are reported only for post-hoc audit; they were not used to choose threshold.'};(artifact/'calibration_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
