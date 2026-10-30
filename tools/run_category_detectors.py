"""Resumable normal-only v2/PatchCore fitting and evaluation for MVTec categories.

Each category writes to a new method/version directory.  ``--resume`` only
skips artifacts that pass safe metadata validation; it never overwrites them.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time, uuid
from dataclasses import fields
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch.utils.data import DataLoader, Subset

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from forgesight_ad.autoencoder_pipeline import MVTecTestDataset, MVTecTrainDataset, refine_predicted_mask
from forgesight_ad.backbone_pipeline import ResNetBackboneConfig, build_resnet_extractor, build_transforms
from forgesight_ad.category_workflows import discover_categories, normal_split, write_json_new
from forgesight_ad.checkpoints import load_checkpoint, validate_feature_statistics_checkpoint
from forgesight_ad.feature_statistics_v2 import METHODOLOGY_ID as V2_ID, fit_statistics
from forgesight_ad.patchcore_pipeline import METHODOLOGY_ID as PC_ID, anomaly_map, greedy_coreset, load_artifact, patch_embeddings, save_artifact
from forgesight_ad.v2_inference import infer_v2, load_v2_artifact


def sha(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()

def atomic_torch_save_new(path: Path, payload: dict) -> None:
    """Publish a new v2 checkpoint without overwriting a pre-existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise FileExistsError(f'Refusing to overwrite existing v2 artifact: {path}')
    temporary=path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        torch.save(payload,temporary)
        try:
            os.link(temporary,path)
        except OSError:
            with path.open('xb'):
                pass
            try:
                os.replace(temporary,path)
            except Exception:
                if path.exists() and path.stat().st_size == 0:path.unlink()
                raise
    except Exception:
        if temporary.exists():temporary.unlink()
        raise
    else:
        if temporary.exists():temporary.unlink()

def config_from_legacy(payload:dict,dataset:Path,output:Path,device:torch.device)->ResNetBackboneConfig:
    values={item.name:payload['config'][item.name] for item in fields(ResNetBackboneConfig) if item.name in payload['config']}
    values.update(project_root=ROOT,dataset_root=dataset,artifacts_root=output,device_name=str(device),batch_size=2,num_workers=0)
    for key in ('feature_layers','feature_blend_weights','score_blend_weights'):
        values[key]=tuple(values[key])
    return ResNetBackboneConfig(**values)

def loaders(dataset:Path,category:str,size:int,fit:list[int],cal:list[int],batch:int=2):
    image_transform,mask_transform=build_transforms(size); train=MVTecTrainDataset(dataset,category,image_transform)
    kw={'batch_size':batch,'shuffle':False,'num_workers':0,'pin_memory':True}
    return train,DataLoader(Subset(train,fit),**kw),DataLoader(Subset(train,cal),**kw),DataLoader(MVTecTestDataset(dataset,category,image_transform,mask_transform,size),**kw)

def metrics(scores,pixels,labels,pixel_labels,threshold,seconds,peak)->dict:
    pred=scores>threshold; normal=labels==0; defect=labels==1
    return {'status':'complete','image_roc_auc':float(roc_auc_score(labels,scores)),'pixel_roc_auc':float(roc_auc_score(pixel_labels.reshape(-1),pixels.reshape(-1))),'image_average_precision':float(average_precision_score(labels,scores)),'pixel_average_precision':float(average_precision_score(pixel_labels.reshape(-1),pixels.reshape(-1))),'fpr':float(pred[normal].mean()),'fnr':float((~pred[defect]).mean()),'confusion_matrix':{'tn':int((~pred[normal]).sum()),'fp':int(pred[normal].sum()),'fn':int((~pred[defect]).sum()),'tp':int(pred[defect].sum())},'evaluation_seconds':seconds,'ms_per_image':seconds*1000/len(labels),'peak_cuda_memory_bytes':peak}

def v2_run(category,dataset,output,seed,device,resume):
    artifact=output/f'resnet50_{category}_v2_seed{seed}'/'model.pt'
    if artifact.exists() and resume:
        payload,cfg,model=load_v2_artifact(artifact,device,'resnet50')
        if payload.get('provenance',{}).get('category')!=category: raise ValueError('artifact category provenance mismatch')
        metrics_path=artifact.parent/'metrics.json'
        if metrics_path.exists(): return 'skipped_verified'
        # A process may stop after immutable fitting but before evaluation. Resume
        # evaluation from the safe artifact; never refit or overwrite it.
        transform,mask_transform=build_transforms(cfg.image_size); test=MVTecTestDataset(dataset,category,transform,mask_transform,cfg.image_size); start=time.perf_counter(); torch.cuda.reset_peak_memory_stats(device); ss=[]; pp=[]; yy=[]; py=[]
        with torch.inference_mode():
            for batch in DataLoader(test,batch_size=2,shuffle=False,num_workers=0,pin_memory=True):
                images=batch['image'].to(device); maps,scores,_=infer_v2(payload,cfg,model,images); ss.extend(scores.cpu().tolist()); pp.extend(maps.cpu().numpy().reshape(-1)); yy.extend(batch['label'].tolist()); py.extend((batch['mask'].numpy().reshape(-1)>0.5).astype(np.uint8))
        result=metrics(np.array(ss),np.array(pp),np.array(yy),np.array(py),float(payload['image_threshold']),time.perf_counter()-start,int(torch.cuda.max_memory_allocated(device)))|payload['provenance']|{'image_threshold':float(payload['image_threshold']),'pixel_threshold':float(payload['pixel_threshold']),'test_images':len(test),'resumed_evaluation':True}; write_json_new(metrics_path,result); return result
    if artifact.exists():raise FileExistsError(artifact)
    legacy=ROOT.parent/'artifacts_resnet_compare'/'resnet50'/category/'resnet50_feature_statistics'/'model.pt'; legacy_payload=load_checkpoint(legacy,device); validate_feature_statistics_checkpoint(legacy_payload,'resnet50')
    destination=artifact.parent; cfg=config_from_legacy(legacy_payload,dataset,destination,device); transform,_=build_transforms(cfg.image_size); train=MVTecTrainDataset(dataset,category,transform); fit,cal=normal_split(len(train),seed)
    _,fit_loader,cal_loader,test_loader=loaders(dataset,category,cfg.image_size,fit,cal); model,source=build_resnet_extractor('resnet50',device,False); start=time.perf_counter(); torch.cuda.reset_peak_memory_stats(device); stats=fit_statistics(model,fit_loader,cfg.feature_layers,cfg.embedding_eps)
    def score(loader,labeled=False):
        ss=[]; pp=[]; yy=[]; py=[]
        with torch.inference_mode():
            for batch in loader:
                images=(batch['image'] if isinstance(batch,dict) else batch[0]).to(device); fake={'statistics':stats,'pixel_threshold':float('inf')}; maps,sc,_=infer_v2(fake,cfg,model,images); ss.extend(sc.cpu().tolist()); pp.extend(maps.cpu().numpy().reshape(-1));
                if labeled:yy.extend(batch['label'].tolist()); py.extend((batch['mask'].numpy().reshape(-1)>0.5).astype(np.uint8))
        return np.array(ss),np.array(pp),np.array(yy),np.array(py)
    cal_scores,cal_pixels,_,_=score(cal_loader); threshold=float(np.quantile(cal_scores,.98,method='higher')); pixel_threshold=float(np.quantile(cal_pixels,.995,method='higher')); payload={'methodology_id':V2_ID,'statistics':stats,'config':cfg.to_dict(),'image_threshold':threshold,'pixel_threshold':pixel_threshold,'provenance':{'methodology_id':V2_ID,'category':category,'backbone':'resnet50','seed':seed,'fit_images':len(fit),'calibration_images':len(cal),'legacy_checkpoint':str(legacy),'legacy_checkpoint_sha256':sha(legacy),'threshold_provenance':'normal-only calibration; test labels excluded'}}
    destination.mkdir(parents=True,exist_ok=False); atomic_torch_save_new(artifact,payload); write_json_new(destination/'split_manifest.json',{'seed':seed,'fit_indices':fit,'calibration_indices':cal,'fit_paths':[str(train.image_paths[i]) for i in fit],'calibration_paths':[str(train.image_paths[i]) for i in cal]})
    scores,pixels,labels,pixel_labels=score(test_loader,True); result=metrics(scores,pixels,labels,pixel_labels,threshold,time.perf_counter()-start,int(torch.cuda.max_memory_allocated(device)))|payload['provenance']|{'image_threshold':threshold,'pixel_threshold':pixel_threshold,'test_images':len(labels)}; write_json_new(destination/'metrics.json',result); return result

def patchcore_run(category,dataset,output,seed,device,resume):
    root=output/f'resnet18_{category}_patchcore_seed{seed}'; artifact=root/'model.pt'; evaluation=root.with_name(root.name+'_evaluation')/'results.json'
    if artifact.exists() and resume:
        payload=load_artifact(artifact,device,'resnet18')
        if payload.get('category') != category: raise ValueError('artifact category provenance mismatch')
        if evaluation.exists():
            try:
                complete=json.loads(evaluation.read_text()).get('status') == 'complete'
            except (OSError,json.JSONDecodeError): complete=False
            if complete:return 'skipped_verified'
            raise ValueError('PatchCore evaluation record exists but is incomplete/corrupt; preserve it and select a new output root')
        # Resume evaluation from the immutable memory bank; do not refit it.
        transform,mask_transform=build_transforms(int(payload['image_size'])); manifest=json.loads((root/'split_manifest.json').read_text()); train=MVTecTrainDataset(dataset,category,transform); model,_=build_resnet_extractor('resnet18',device,False); layers=tuple(payload['feature_layers']); bank=payload['memory_bank']
        def resumed_score(loader,labeled=False):
            ss=[]; maps=[]; yy=[]; py=[]
            with torch.inference_mode():
                for batch in loader:
                    images=(batch['image'] if isinstance(batch,dict) else batch[0]).to(device); amap=anomaly_map(model,images,bank,layers); ss.extend(amap.flatten(1).max(1).values.cpu().tolist()); maps.extend(amap.cpu().numpy().reshape(-1));
                    if labeled: yy.extend(batch['label'].tolist()); py.extend((batch['mask'].numpy().reshape(-1)>0.5).astype(np.uint8))
            return np.array(ss),np.array(maps),np.array(yy),np.array(py)
        cal_scores,_,_,_=resumed_score(DataLoader(Subset(train,manifest['calibration_indices']),batch_size=2,shuffle=False)); threshold=float(np.quantile(cal_scores,.98,method='higher')); test=MVTecTestDataset(dataset,category,transform,mask_transform,int(payload['image_size'])); torch.cuda.reset_peak_memory_stats(device); start=time.perf_counter(); scores,pixels,labels,pixel_labels=resumed_score(DataLoader(test,batch_size=2,shuffle=False),True); result=metrics(scores,pixels,labels,pixel_labels,threshold,time.perf_counter()-start,int(torch.cuda.max_memory_allocated(device)))|{'methodology_id':PC_ID,'category':category,'backbone':'resnet18','seed':seed,'fit_images':payload['fit_images'],'calibration_images':len(manifest['calibration_indices']),'test_images':len(test),'threshold':threshold,'memory_bank_shape':list(bank.shape),'resumed_evaluation':True}; evaluation.parent.mkdir(parents=True,exist_ok=False); write_json_new(evaluation,result); return result
    if artifact.exists():raise FileExistsError(artifact)
    transform,mask_transform=build_transforms(512); train=MVTecTrainDataset(dataset,category,transform); manifest_path=root/'split_manifest.json'
    if root.exists():
        if not resume or not manifest_path.is_file():
            raise FileExistsError(f'Incomplete PatchCore directory requires --resume and a split manifest: {root}')
        manifest=json.loads(manifest_path.read_text())
        fit,cal=manifest['fit_indices'],manifest['calibration_indices']
        if set(fit).intersection(cal) or sorted(fit+cal)!=list(range(len(train))):
            raise ValueError('Existing PatchCore split manifest is invalid')
    else:
        root.mkdir(parents=True,exist_ok=False); fit,cal=normal_split(len(train),seed)
        manifest={'seed':seed,'fit_indices':fit,'calibration_indices':cal,'fit_paths':[str(train.image_paths[i]) for i in fit],'calibration_paths':[str(train.image_paths[i]) for i in cal]}
        write_json_new(manifest_path,manifest)
    model,source=build_resnet_extractor('resnet18',device,False); layers=('layer2','layer3'); torch.cuda.reset_peak_memory_stats(device); start=time.perf_counter(); chunks=[]
    with torch.inference_mode():
        for images,_ in DataLoader(Subset(train,fit),batch_size=2,shuffle=False,num_workers=0,pin_memory=True):
            patches,_=patch_embeddings(model(images.to(device)),layers); chunks.append(patches.cpu())
    patches=torch.cat(chunks); count=min(8192,len(patches)); candidates=patches[torch.linspace(0,len(patches)-1,count).long()].to(device); bank=candidates[greedy_coreset(candidates,256,seed)].cpu().contiguous(); payload={'methodology_id':PC_ID,'category':category,'backbone_name':'resnet18','backbone_source':source,'feature_layers':list(layers),'image_size':512,'embedding_normalization':'l2','memory_bank':bank,'coreset':{'algorithm':'deterministic_greedy_kcenter_on_uniform_candidate_subset','candidate_count':count,'selected_count':len(bank)},'seed':seed,'split_manifest_sha256':sha(manifest_path),'calibration':{'status':'normal_only','indices':cal},'fit_images':len(fit),'total_patches':len(patches),'fit_seconds':time.perf_counter()-start,'peak_cuda_memory_bytes':int(torch.cuda.max_memory_allocated(device))}; save_artifact(artifact,payload)
    def score(loader,labeled=False):
        ss=[]; maps=[]; yy=[]; py=[]
        with torch.inference_mode():
            for batch in loader:
                images=(batch['image'] if isinstance(batch,dict) else batch[0]).to(device); amap=anomaly_map(model,images,bank,layers); ss.extend(amap.flatten(1).max(1).values.cpu().tolist()); maps.extend(amap.cpu().numpy().reshape(-1));
                if labeled:yy.extend(batch['label'].tolist());py.extend((batch['mask'].numpy().reshape(-1)>0.5).astype(np.uint8))
        return np.array(ss),np.array(maps),np.array(yy),np.array(py)
    cal_scores,_,_,_=score(DataLoader(Subset(train,cal),batch_size=2,shuffle=False)); threshold=float(np.quantile(cal_scores,.98,method='higher')); test=MVTecTestDataset(dataset,category,transform,mask_transform,512); torch.cuda.reset_peak_memory_stats(device); estart=time.perf_counter(); scores,pixels,labels,pixel_labels=score(DataLoader(test,batch_size=2,shuffle=False),True); result=metrics(scores,pixels,labels,pixel_labels,threshold,time.perf_counter()-estart,int(torch.cuda.max_memory_allocated(device)))|{'methodology_id':PC_ID,'category':category,'backbone':'resnet18','seed':seed,'fit_images':len(fit),'calibration_images':len(cal),'test_images':len(labels),'threshold':threshold,'memory_bank_shape':list(bank.shape)}; evaluation.parent.mkdir(parents=True,exist_ok=False); write_json_new(evaluation,result); return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--method',choices=('v2','patchcore','both'),default='both'); p.add_argument('--category'); p.add_argument('--all-categories',action='store_true'); p.add_argument('--resume',action='store_true'); p.add_argument('--device',default='cuda'); p.add_argument('--seed',type=int,default=42); p.add_argument('--dataset-root',type=Path,default=ROOT.parent/'mvtec-ad'/'versions'/'2'); p.add_argument('--output-root',type=Path,default=ROOT/'artifacts'/'category_experiments'); a=p.parse_args(); device=torch.device('cuda:0' if a.device in ('cuda','cuda:0') else a.device); cats=discover_categories(a.dataset_root) if a.all_categories else [a.category];
    if not cats or any(not c for c in cats):p.error('provide --category or --all-categories')
    for category in cats:
        print(f'category:{category}:begin',flush=True)
        existing_v2=ROOT/'artifacts'/'v2_experiments'/f'resnet50_{category}_v2_seed{a.seed}'/'model.pt'
        existing_pc=ROOT/'artifacts'/'patchcore_experiments'/f'resnet18_{category}_seed{a.seed}'/'model.pt'
        if a.method in ('v2','both'):
            value='skipped_existing_validated_artifact' if existing_v2.exists() else v2_run(category,a.dataset_root,a.output_root/'v2',a.seed,device,a.resume)
            print(json.dumps({'category':category,'v2':value},default=str),flush=True)
        if a.method in ('patchcore','both'):
            value='skipped_existing_validated_artifact' if existing_pc.exists() else patchcore_run(category,a.dataset_root,a.output_root/'patchcore',a.seed,device,a.resume)
            print(json.dumps({'category':category,'patchcore':value},default=str),flush=True)
if __name__=='__main__':main()
