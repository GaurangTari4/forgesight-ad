"""Read-only calibration and post-hoc failure diagnostics for completed artifacts."""
from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
from torch.utils.data import DataLoader,Subset
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from forgesight_ad.autoencoder_pipeline import MVTecTrainDataset,MVTecTestDataset
from forgesight_ad.backbone_pipeline import build_resnet_extractor,build_transforms
from forgesight_ad.category_workflows import discover_categories,write_json_new
from forgesight_ad.patchcore_pipeline import anomaly_map,load_artifact
from forgesight_ad.v2_inference import infer_v2,load_v2_artifact

DATA=ROOT.parent/'mvtec-ad'/'versions'/'2'
def artifact(method,category):
 if method=='v2':
  root=ROOT/'artifacts'/'v2_experiments'/f'resnet50_{category}_v2_seed42' if category=='bottle' else ROOT/'artifacts'/'category_experiments'/'v2'/f'resnet50_{category}_v2_seed42';return root/'model.pt',root/'split_manifest.json',root/'metrics.json'
 root=ROOT/'artifacts'/'patchcore_experiments'/f'resnet18_{category}_seed42' if category=='bottle' else ROOT/'artifacts'/'category_experiments'/'patchcore'/f'resnet18_{category}_patchcore_seed42';manifest=ROOT/'artifacts'/'v2_experiments'/'resnet50_bottle_v2_seed42'/'split_manifest.json' if category=='bottle' else root/'split_manifest.json';return root/'model.pt',manifest,root.with_name(root.name+'_evaluation')/'results.json'
def summary(x):
 x=np.asarray(x,float);return {'count':len(x),'mean':float(x.mean()),'std':float(x.std()),'min':float(x.min()),'p50':float(np.quantile(x,.5)),'p90':float(np.quantile(x,.9)),'p95':float(np.quantile(x,.95)),'p98':float(np.quantile(x,.98)),'max':float(x.max()),'upper_tail_outliers':int((x>np.quantile(x,.95)).sum())}
def wilson(k,n):
 z=1.96;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/d;return [float(c-h),float(c+h)]
def visual(out,path,image,amap,mask):
 heat=(np.clip(amap/(amap.max()+1e-12),0,1)*255).astype(np.uint8);Image.fromarray(np.asarray(image.convert('RGB').resize((amap.shape[1],amap.shape[0])))).save(out/'false_negative_input.png');Image.fromarray(heat).save(out/'false_negative_heatmap.png');Image.fromarray((mask>0.5).astype(np.uint8)*255).save(out/'false_negative_ground_truth.png');(out/'false_negative_source.txt').write_text(str(path),encoding='utf-8')
def run(method,category,device):
 ap,mp,rp=artifact(method,category);stored=json.loads(rp.read_text());manifest=json.loads(mp.read_text());transform,mask_transform=build_transforms(512);train=MVTecTrainDataset(DATA,category,transform);cal=manifest['calibration_indices'];test=MVTecTestDataset(DATA,category,transform,mask_transform,512)
 if method=='v2':
  payload,cfg,model=load_v2_artifact(ap,device,'resnet50');threshold=float(payload['image_threshold'])
  def score(images):
   maps,scores,_=infer_v2(payload,cfg,model,images);return maps,scores
 else:
  payload=load_artifact(ap,device,'resnet18');model,_=build_resnet_extractor('resnet18',device,False);threshold=float(stored['threshold']);layers=tuple(payload['feature_layers']);bank=payload['memory_bank']
  def score(images):
   maps=anomaly_map(model,images,bank,layers);return maps,maps.flatten(1).max(1).values
 def collect(loader,labeled=False,out=None):
  scores=[];labels=[];types=[];false=[]
  with torch.inference_mode():
   for batch in loader:
    images=(batch['image'] if isinstance(batch,dict) else batch[0]).to(device);maps,sc=score(images);scores.extend(sc.cpu().tolist())
    if labeled:
     for i in range(len(sc)):
      y=int(batch['label'][i]);typ=batch['defect_type'][i];labels.append(y);types.append(typ)
      if y and float(sc[i])<=threshold: false.append((typ,batch['path'][i],maps[i,0].cpu().numpy(),batch['mask'][i,0].numpy()))
  return np.array(scores),np.array(labels),types,false
 cal_scores,_,_,_=collect(DataLoader(Subset(train,cal),batch_size=2,shuffle=False));normal_scores,normal_labels,_,_=collect(DataLoader(test,batch_size=2,shuffle=False),True);test_scores,labels,types,false=collect(DataLoader(test,batch_size=2,shuffle=False),True)
 # normal_scores above contains all test scores; filter by labels below.
 normal=test_scores[labels==0];defect=test_scores[labels==1];recomputed=float(np.quantile(cal_scores,.98,method='higher'));subtypes={}
 for typ,_,_,_ in false:subtypes[typ]=subtypes.get(typ,0)+1
 fn=len(false);fp=int(((normal)>threshold).sum());delta=abs(recomputed-threshold);result={'method':method,'category':category,'artifact':str(ap),'threshold':threshold,'threshold_absolute_delta':delta,'stored_threshold_consistent_within_1e4_relative':bool(delta<=max(1e-4,abs(threshold)*1e-4)),'recomputed_calibration_q98':recomputed,'calibration_scores':cal_scores.tolist(),'calibration_summary':summary(cal_scores),'normal_test_summary':summary(normal),'defect_test_summary':summary(defect),'post_hoc_test_errors':{'false_negatives':fn,'false_positives':fp,'false_negative_by_defect_type':subtypes,'fn_wilson_95':wilson(fn,len(defect)),'fp_wilson_95':wilson(fp,len(normal))},'notes':'Post-hoc test analysis only; no thresholds were changed or selected using test scores.'}
 out=ROOT/'artifacts'/'failure_analysis'/method/category;out.mkdir(parents=True,exist_ok=False);write_json_new(out/'diagnostics.json',result)
 if false:
  typ,path,amap,mask=false[0];visual(out,path,Image.open(path),amap,mask)
 return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--method',choices=('v2','patchcore','both'),default='both');p.add_argument('--category');p.add_argument('--all-categories',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--device',default='cuda:0');a=p.parse_args();cats=discover_categories(DATA) if a.all_categories else [a.category];d=torch.device(a.device)
 for c in cats:
  for m in (('v2','patchcore') if a.method=='both' else (a.method,)):
   target=ROOT/'artifacts'/'failure_analysis'/m/c/'diagnostics.json'
   if target.exists() and a.resume:print(f'{m}:{c}:skipped',flush=True);continue
   print(json.dumps(run(m,c,d),default=str),flush=True)
if __name__=='__main__':main()
