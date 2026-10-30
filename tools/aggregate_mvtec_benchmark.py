"""Read-only audit and aggregation of completed v2 and PatchCore category runs."""
from __future__ import annotations
import csv, hashlib, json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from forgesight_ad.category_workflows import discover_categories, macro_average
from forgesight_ad.feature_statistics_v2 import METHODOLOGY_ID as V2_ID
from forgesight_ad.patchcore_pipeline import METHODOLOGY_ID as PC_ID, validate_artifact

DATA=ROOT.parent/'mvtec-ad'/'versions'/'2'
METRICS=('image_roc_auc','pixel_roc_auc','image_average_precision','pixel_average_precision','fpr','fnr')

def digest(path):
 h=hashlib.sha256();
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def v2_record(category):
 root=ROOT/'artifacts'/'v2_experiments'/f'resnet50_{category}_v2_seed42' if category=='bottle' else ROOT/'artifacts'/'category_experiments'/'v2'/f'resnet50_{category}_v2_seed42'
 artifact=root/'model.pt'; metrics=root/'metrics.json'; manifest=root/'split_manifest.json'
 payload=torch.load(artifact,map_location='cpu',weights_only=True); result=read(metrics); split=read(manifest)
 if payload.get('methodology_id')!=V2_ID or payload.get('provenance',{}).get('category')!=category or result.get('category')!=category or result.get('backbone')!='resnet50':raise ValueError('v2 provenance mismatch')
 if set(split['fit_indices'])&set(split['calibration_indices']):raise ValueError('v2 split overlap')
 if not all(key in payload for key in ('statistics','image_threshold','pixel_threshold')):raise ValueError('v2 missing threshold/statistics')
 if 'confusion_matrix' not in result:
  normal=len(list((DATA/category/'test'/'good').glob('*'))); defect=result['test_images']-normal; fp=round(float(result['fpr'])*normal); fn=round(float(result['fnr'])*defect); result['confusion_matrix']={'tn':normal-fp,'fp':fp,'fn':fn,'tp':defect-fn}
 return {**result,'method':'v2','artifact':str(artifact),'manifest':str(manifest),'artifact_sha256':digest(artifact),'calibration_distribution_available':'calibration_scores' in result}

def pc_record(category):
 root=ROOT/'artifacts'/'patchcore_experiments'/f'resnet18_{category}_seed42' if category=='bottle' else ROOT/'artifacts'/'category_experiments'/'patchcore'/f'resnet18_{category}_patchcore_seed42'
 artifact=root/'model.pt'; metrics=root.with_name(root.name+'_evaluation')/'results.json'; manifest=(ROOT/'artifacts'/'v2_experiments'/'resnet50_bottle_v2_seed42'/'split_manifest.json') if category=='bottle' else root/'split_manifest.json'
 payload=torch.load(artifact,map_location='cpu',weights_only=True);validate_artifact(payload,'resnet18'); result=read(metrics); split=read(manifest)
 if payload.get('category','bottle')!=category or result.get('category','bottle')!=category or result.get('backbone','resnet18')!='resnet18':raise ValueError('PatchCore provenance mismatch')
 result.setdefault('category',category);result.setdefault('backbone','resnet18');result.setdefault('status','complete')
 if payload.get('split_manifest_sha256')!=digest(manifest) or set(split['fit_indices'])&set(split['calibration_indices']):raise ValueError('PatchCore split manifest mismatch')
 if 'threshold' not in result:raise ValueError('PatchCore threshold missing')
 return {**result,'method':'patchcore','artifact':str(artifact),'manifest':str(manifest),'artifact_sha256':digest(artifact),'calibration_distribution_available':'calibration_scores' in result}

def pooled(rows):
 cm={k:sum(r['confusion_matrix'][k] for r in rows) for k in ('tn','fp','fn','tp')}
 return {'confusion_matrix':cm,'pooled_fpr':cm['fp']/(cm['fp']+cm['tn']),'pooled_fnr':cm['fn']/(cm['fn']+cm['tp'])}

def chart(rows,out,metric):
 cats=sorted({r['category'] for r in rows});methods=('v2','patchcore');x=range(len(cats));w=.38
 fig,ax=plt.subplots(figsize=(15,5));
 for i,m in enumerate(methods):ax.bar([v+(i-.5)*w for v in x],[next(r[metric] for r in rows if r['method']==m and r['category']==c) for c in cats],w,label=m)
 ax.set_xticks(list(x),cats,rotation=45,ha='right');ax.set_ylim(0,1);ax.set_ylabel(metric);ax.legend();fig.tight_layout();fig.savefig(out,dpi=160);plt.close(fig)

def main():
 categories=discover_categories(DATA);rows=[];invalid=[]
 for category in categories:
  for method,loader in (('v2',v2_record),('patchcore',pc_record)):
   try: rows.append(loader(category))
   except Exception as exc:invalid.append({'category':category,'method':method,'error':str(exc)})
 if invalid:raise RuntimeError(json.dumps({'invalid':invalid},indent=2))
 out=ROOT/'artifacts'/'benchmark_summary'
 if out.exists() and any(out.iterdir()):
  # Recovery is permitted only for the known header-only CSV left by an
  # interrupted aggregation before any scientific result was published.
  entries=list(out.iterdir())
  if not (len(entries)==1 and entries[0].name=='mvtec_15_categories.csv'):raise FileExistsError(f'Refusing to overwrite benchmark summary: {out}')
 out.mkdir(parents=True,exist_ok=True)
 fields=['method','category',*METRICS,'fit_images','calibration_images','test_images','evaluation_seconds','ms_per_image','peak_cuda_memory_bytes','artifact','manifest','artifact_sha256','calibration_distribution_available']
 with (out/'mvtec_15_categories.csv').open('w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
 summaries={m:{'macro':macro_average([r for r in rows if r['method']==m]),**pooled([r for r in rows if r['method']==m])} for m in ('v2','patchcore')}
 for m in summaries:
  mr=[r for r in rows if r['method']==m]
  summaries[m]['best_worst']={key:{'best':max(mr,key=lambda r:r[key])['category'],'worst':min(mr,key=lambda r:r[key])['category']} for key in ('image_roc_auc','pixel_roc_auc','image_average_precision','pixel_average_precision')}
  summaries[m]['missed_defects']=sorted(((r['category'],r['confusion_matrix']['fn']) for r in mr),key=lambda x:x[1],reverse=True)
  summaries[m]['false_alarms']=sorted(((r['category'],r['confusion_matrix']['fp']) for r in mr),key=lambda x:x[1],reverse=True)
  summaries[m]['below_auroc_targets']=[r['category'] for r in mr if r['image_roc_auc']<.95 or r['pixel_roc_auc']<.95]
 payload={'categories':categories,'rows':rows,'summaries':summaries,'invalid':invalid};(out/'mvtec_15_categories.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
 for metric in ('image_roc_auc','pixel_roc_auc','fpr','fnr'):chart(rows,out/f'{metric}.png',metric)
 lines=['# MVTec AD 15-category benchmark','','All metrics are read-only aggregations of completed normal-only calibration runs. Test labels were used only for evaluation.','']
 for m,s in summaries.items():
  lines+= [f'## {m}', '', f"Macro image/pixel AUROC: {s['macro']['image_roc_auc']:.6f} / {s['macro']['pixel_roc_auc']:.6f}. Pooled FPR/FNR: {s['pooled_fpr']:.6f} / {s['pooled_fnr']:.6f}.", '', '| category | image AUROC | pixel AUROC | image AP | pixel AP | FPR | FNR |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
  lines += [f"| {r['category']} | {r['image_roc_auc']:.6f} | {r['pixel_roc_auc']:.6f} | {r['image_average_precision']:.6f} | {r['pixel_average_precision']:.6f} | {r['fpr']:.6f} | {r['fnr']:.6f} |" for r in rows if r['method']==m]
 lines += ['', '## Methodology limitations', '', 'v2 uses a frozen ResNet-50 and unweighted fitted layer statistics combined by configured score weights. PatchCore uses a frozen ResNet-18, normalized layer2/layer3 patch embeddings, and nearest-neighbor distances to a deterministic coreset. Backbone and scoring differ, so this is not an isolated algorithm-only comparison. Results are exploratory MVTec test-set evidence, not production reliability. Thresholds were calibrated from disjoint normal-only training splits; no threshold was selected from test outcomes. Generic per-run records do not persist individual calibration score distributions, so post-hoc distribution quantiles cannot be reconstructed without rescoring; thresholds and provenance are retained.']
 (ROOT/'docs'/'MVTEC_15_CATEGORY_BENCHMARK.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
