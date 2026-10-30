"""Prespecified, isolated representation diagnostics; not test-set model selection."""
from __future__ import annotations
import argparse,json,sys,time
from datetime import datetime,timezone
from pathlib import Path
import torch
from PIL import Image,ImageDraw
from torch.utils.data import DataLoader,Subset
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from forgesight_ad.autoencoder_pipeline import MVTecTrainDataset
from forgesight_ad.backbone_pipeline import build_transforms
from forgesight_ad.v2_inference import infer_v2,load_v2_artifact

def synthetic(image:Image.Image,kind:str,seed:int)->tuple[Image.Image,tuple[int,int,int,int]]:
 """Deterministic synthetic proxy; returns image and known modified region."""
 im=image.convert('RGB').resize((512,512));draw=ImageDraw.Draw(im);g=torch.Generator().manual_seed(seed);x=int(torch.randint(120,350,(1,),generator=g));y=int(torch.randint(120,350,(1,),generator=g))
 if kind=='small_local':box=(x,y,x+12,y+12);draw.rectangle(box,fill=(255,20,20))
 elif kind=='low_contrast':box=(x,y,x+40,y+40);draw.rectangle(box,fill=(185,185,185))
 elif kind=='thin_scratch':box=(x,y,x+80,y+3);draw.line((x,y,x+80,y+3),fill=(245,245,245),width=2)
 elif kind=='color_shift':box=(x,y,x+48,y+48);draw.rectangle(box,fill=(30,180,210))
 elif kind=='localized_change':box=(x,y,x+64,y+64);draw.ellipse(box,fill=(20,20,20))
 else:raise ValueError(kind)
 return im,box
def main():
 p=argparse.ArgumentParser();p.add_argument('--category',default='bottle');p.add_argument('--samples',type=int,default=10);p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cuda:0');a=p.parse_args();artifact=ROOT/'artifacts'/'v2_experiments'/f'resnet50_{a.category}_v2_seed42'/'model.pt';payload,cfg,model=load_v2_artifact(artifact,torch.device(a.device),'resnet50');manifest=json.loads((artifact.parent/'split_manifest.json').read_text());transform,_=build_transforms(cfg.image_size);train=MVTecTrainDataset(ROOT.parent/'mvtec-ad'/'versions'/'2',a.category,transform);indices=manifest['calibration_indices'][:a.samples];out=ROOT/'artifacts'/'representation_experiments'/f'{a.category}_v2_synthetic_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}';out.mkdir(parents=True,exist_ok=False);records=[]
 for i in indices:
  source=Image.open(train.image_paths[i]).convert('RGB');baseline=transform(source).unsqueeze(0).to(a.device)
  with torch.inference_mode():_,base_score,_=infer_v2(payload,cfg,model,baseline)
  for kind in json.loads((ROOT/'configs'/'representation_experiments.json').read_text())['synthetic_anomalies']:
   altered,box=synthetic(source,kind,a.seed+i);tensor=transform(altered).unsqueeze(0).to(a.device)
   with torch.inference_mode():amap,score,_=infer_v2(payload,cfg,model,tensor)
   x0,y0,x1,y1=box;region=amap[0,0,y0:y1+1,x0:x1+1];records.append({'source_index':i,'source_path':str(train.image_paths[i]),'synthetic_type':kind,'box':box,'baseline_score':float(base_score),'synthetic_score':float(score),'score_delta':float(score-base_score),'region_max':float(region.max()),'map_max':float(amap.max())})
 (out/'synthetic_diagnostics.json').write_text(json.dumps({'protocol':'diagnostic proxy only; no test labels; no model selection','category':a.category,'seed':a.seed,'samples':len(indices),'records':records},indent=2),encoding='utf-8');print(json.dumps({'output':str(out),'records':len(records)},indent=2))
if __name__=='__main__':main()
