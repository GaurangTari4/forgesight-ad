"""Standalone PatchCore visual smoke validation; does not import Gradio."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));OUT=ROOT/'validation_runs'/'patchcore_visuals'/'run_20261030'
from forgesight_ad.patchcore_pipeline import load_artifact,anomaly_map
from forgesight_ad.backbone_pipeline import build_resnet_extractor,build_transforms
def main():
 OUT.mkdir(parents=True,exist_ok=False);d=torch.device('cuda:0');art=ROOT/'artifacts/patchcore_experiments/resnet18_bottle_seed42/model.pt';a=load_artifact(art,d,'resnet18');e,_=build_resnet_extractor('resnet18',d,False);t,_=build_transforms(512);data=ROOT.parent/'mvtec-ad/versions/2/bottle/test';report={'artifact':str(art),'device':str(d),'threshold':.9641508460044861,'samples':{}}
 for name,path in {'normal':data/'good/000.png','defective':data/'broken_large/000.png'}.items():
  im=Image.open(path).convert('RGB').resize((512,512));x=t(im).unsqueeze(0).to(d);m=anomaly_map(e,x,a['memory_bank'],tuple(a['feature_layers']));z=m.squeeze().cpu().numpy();score=float(z.max());heat=(np.clip(z/(z.max()+1e-8),0,1)*255).astype(np.uint8);mask=(z>report['threshold']).astype(np.uint8)*255;Image.fromarray(np.asarray(im)).save(OUT/f'{name}_input.png');Image.fromarray(heat).save(OUT/f'{name}_heatmap.png');Image.fromarray(mask).save(OUT/f'{name}_mask.png');report['samples'][name]={'path':str(path),'score':score,'prediction':'fail' if score>report['threshold'] else 'pass','map_shape':list(m.shape),'finite':bool(np.isfinite(z).all()),'image_size':list(im.size)}
 (OUT/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
