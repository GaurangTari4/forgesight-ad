"""Read-only bootstrap study of saved normal-only calibration score arrays."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

def q98(values:np.ndarray)->float:return float(np.quantile(values,.98,method='higher'))
def bootstrap_quantile(values,seed=42,repeats=5000):
 x=np.asarray(values,float)
 if len(x)<2:raise ValueError('at least two calibration scores required')
 rng=np.random.default_rng(seed);samples=rng.choice(x,size=(repeats,len(x)),replace=True);est=np.quantile(samples,.98,axis=1,method='higher')
 return {'seed':seed,'repeats':repeats,'mean':float(est.mean()),'std':float(est.std()),'ci95':[float(np.quantile(est,.025)),float(np.quantile(est,.975))],'relative_ci_width':float((np.quantile(est,.975)-np.quantile(est,.025))/(abs(q98(x))+1e-12))}
def robust_estimators(values):
 x=np.asarray(values,float);q1,q3=np.quantile(x,[.25,.75]);median=float(np.median(x));mad=float(np.median(np.abs(x-median)))
 return {'empirical_q98':q98(x),'tukey_upper_fence':float(q3+1.5*(q3-q1)),'median_plus_3_mad_scaled':float(median+3*1.4826*mad),'normal_mean_plus_2_326_std':float(x.mean()+2.326*x.std())}
def provenance(method,category,data):
 if data.get('method')!=method or data.get('category')!=category or not Path(data['artifact']).is_file():raise ValueError('diagnostic provenance mismatch')
 manifest=Path(data['artifact']).parent/'split_manifest.json'
 if category=='bottle' and method=='patchcore':manifest=ROOT/'artifacts'/'v2_experiments'/'resnet50_bottle_v2_seed42'/'split_manifest.json'
 split=json.loads(manifest.read_text());return len(split['calibration_indices'])
def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);p.add_argument('--repeats',type=int,default=5000);a=p.parse_args();out=ROOT/'artifacts'/'calibration_stability';out.mkdir(parents=True,exist_ok=False);rows=[]
 for method in ('v2','patchcore'):
  for source in sorted((ROOT/'artifacts'/'failure_analysis'/method).glob('*/diagnostics.json')):
   data=json.loads(source.read_text());category=data['category'];n=provenance(method,category,data);scores=np.asarray(data['calibration_scores'],float)
   if len(scores)!=n:raise ValueError(f'calibration count mismatch: {method}/{category}')
   boot=bootstrap_quantile(scores,a.seed,a.repeats);robust=robust_estimators(scores);rows.append({'method':method,'category':category,'calibration_count':n,'stored_threshold':data['threshold'],'stored_threshold_delta':data.get('threshold_absolute_delta'),'summary':data['calibration_summary'],'bootstrap_q98':boot,'alternative_normal_only_estimators':robust,'unstable_upper_tail':bool(boot['relative_ci_width']>.1 or data['calibration_summary']['upper_tail_outliers']>=3),'notes':'Alternatives are descriptive/prespecified candidates only; no test-set optimization was performed.'})
 (out/'stability_metrics.json').write_text(json.dumps(rows,indent=2,allow_nan=False),encoding='utf-8')
 fig,ax=plt.subplots(figsize=(14,5));
 for method in ('v2','patchcore'):
  r=[x for x in rows if x['method']==method];ax.plot([x['category'] for x in r],[x['bootstrap_q98']['relative_ci_width'] for x in r],marker='o',label=method)
 ax.axhline(.1,color='r',linestyle='--',label='10% relative CI width flag');ax.set_ylabel('bootstrap q98 relative CI width');ax.tick_params(axis='x',rotation=45);ax.legend();fig.tight_layout();fig.savefig(out/'q98_relative_ci_width.png',dpi=160);plt.close(fig)
 lines=['# Calibration stability study','','This is a read-only bootstrap analysis of normal-only calibration scores. Bootstrap intervals quantify sampling variability, not operational FPR guarantees. Alternative estimators are not selected using MVTec defect-test outcomes.','',f'Bootstrap seed: `{a.seed}`; replicates: `{a.repeats}`.','', '| method | unstable upper-tail categories | median calibration n |', '| --- | --- | ---: |']
 for method in ('v2','patchcore'):
  r=[x for x in rows if x['method']==method];unstable=[x['category'] for x in r if x['unstable_upper_tail']];lines.append(f"| {method} | {', '.join(unstable) or 'none'} | {int(np.median([x['calibration_count'] for x in r]))} |")
 lines += ['','## Interpretation','','The empirical q98 is determined by one or a few upper-tail normal images at these sample sizes. Tukey fences, MAD-based thresholds, and normal-model thresholds make different distributional assumptions; they are future prespecified candidates, not replacements for stored thresholds. Any operational comparison requires fresh independent calibration/development data and an untouched final evaluation set.','','## Data requirements','','To demonstrate an FPR below 2% with 95% confidence and zero observed false positives, an independent normal set requires approximately 184 samples (`1 - 0.05^(1/n) <= 0.02`). Demonstrating an FNR below 0.5% with zero misses similarly needs approximately 598 independent defective samples. These are binomial planning approximations; subtype coverage can require substantially more data.','','## Research roadmap','','1. Collect independent normal calibration pools per category with sufficient tail coverage.','2. Prespecify one empirical and one robust threshold estimator before a development comparison.','3. Use a separate development set to choose methodology, then lock it.','4. Evaluate once on an untouched, subtype-stratified test set with Wilson intervals.','5. Persist score arrays and manifests as immutable diagnostic sidecars.']
 (ROOT/'docs'/'CALIBRATION_STABILITY_STUDY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps({'records':len(rows),'unstable':[f"{x['method']}/{x['category']}" for x in rows if x['unstable_upper_tail']],'output':str(out)},indent=2))
if __name__=='__main__':main()
