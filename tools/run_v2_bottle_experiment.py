"""Isolated normal-only ResNet-50/bottle feature-statistics-v2 experiment."""
from __future__ import annotations
import argparse, hashlib, json, random, sys, time
from dataclasses import fields
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]; WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
from forgesight_ad.autoencoder_pipeline import MVTecTestDataset, MVTecTrainDataset, gaussian_blur, refine_predicted_mask, topk_mean_scores
from forgesight_ad.backbone_pipeline import ResNetBackboneConfig, build_resnet_extractor, build_transforms, estimate_foreground_mask
from forgesight_ad.checkpoints import load_checkpoint, validate_feature_statistics_checkpoint
from forgesight_ad.feature_statistics_v2 import METHODOLOGY_ID, fit_statistics, score


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def cfg(payload, dataset, output, device):
    values={x.name:payload['config'][x.name] for x in fields(ResNetBackboneConfig) if x.name in payload['config']}
    values.update(project_root=ROOT,dataset_root=dataset,artifacts_root=output,device_name=str(device),batch_size=2,num_workers=0)
    for name in ('feature_layers','feature_blend_weights','score_blend_weights'): values[name]=tuple(values[name])
    return ResNetBackboneConfig(**values)

def postprocess(raw, images, config):
    result=gaussian_blur(raw, config.heatmap_blur_kernel, config.heatmap_blur_sigma)
    pad=(images.max(1,keepdim=True).values <= (0.5/255)).float()
    if torch.any(pad>0): result=result*(1-F.max_pool2d(pad,15,1,7))
    if config.suppress_background_anomalies: result=result*estimate_foreground_mask(images,float(config.background_tolerance),float(config.background_min_coverage))
    margin=max(12,images.shape[-1]//12); result[...,:margin,:]=0; result[...,-margin:,:]=0; result[...,:,:margin]=0; result[...,:,-margin:]=0
    return result, topk_mean_scores(result, config.top_k_ratio)

def maps(extractor, images, stats, config): return postprocess(score(extractor,images,stats,config.feature_layers,config.score_blend_weights,config.embedding_eps),images,config)

def scores(loader, extractor, stats, config, labels=False):
    image=[]; pixels=[]; y=[]; py=[]
    with torch.inference_mode():
      for batch in loader:
        ims=(batch['image'] if isinstance(batch,dict) else batch[0]).to(config.device,non_blocking=True); heat,sc=maps(extractor,ims,stats,config)
        image.append(sc.cpu().numpy()); pixels.append(heat.cpu().numpy().reshape(-1))
        if labels: y.append(batch['label'].numpy()); py.append((batch['mask'].numpy().reshape(-1)>0.5).astype(np.uint8))
    return np.concatenate(image),np.concatenate(pixels), (np.concatenate(y) if y else None), (np.concatenate(py) if py else None)

def main():
 p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,default=42); p.add_argument('--calibration-fraction',type=float,default=.2); p.add_argument('--output',type=Path,default=ROOT/'artifacts'/'v2_experiments'/'resnet50_bottle_v2_seed42'); a=p.parse_args()
 dataset=WORKSPACE/'mvtec-ad'/'versions'/'2'; legacy=WORKSPACE/'artifacts_resnet_compare'/'resnet50'/'bottle'/'resnet50_feature_statistics'/'model.pt'; device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 if device.type!='cuda': raise RuntimeError('CUDA is required');
 if a.output.exists(): raise FileExistsError(f'Output exists: {a.output}')
 payload=load_checkpoint(legacy,device); validate_feature_statistics_checkpoint(payload,'resnet50'); config=cfg(payload,dataset,a.output,device); transform,mask_transform=build_transforms(config.image_size)
 train=MVTecTrainDataset(dataset,'bottle',transform=transform); indices=list(range(len(train))); random.Random(a.seed).shuffle(indices); calibration_count=max(1,round(len(indices)*a.calibration_fraction)); calibration_indices=sorted(indices[:calibration_count]); fitting_indices=sorted(indices[calibration_count:])
 extractor,source=build_resnet_extractor('resnet50',device,allow_random_init=False); extractor.eval(); fit_loader=DataLoader(Subset(train,fitting_indices),batch_size=2,shuffle=False,num_workers=0,pin_memory=True); cal_loader=DataLoader(Subset(train,calibration_indices),batch_size=2,shuffle=False,num_workers=0,pin_memory=True)
 torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device); start=time.perf_counter(); stats=fit_statistics(extractor,fit_loader,config.feature_layers,config.embedding_eps); calibration_image,calibration_pixel,_,_=scores(cal_loader,extractor,stats,config); image_threshold=float(np.quantile(calibration_image,.98,method='higher')); pixel_threshold=float(np.quantile(calibration_pixel,.995,method='higher'))
 test=MVTecTestDataset(dataset,'bottle',transform,mask_transform,config.image_size); test_loader=DataLoader(test,batch_size=2,shuffle=False,num_workers=0,pin_memory=True); cuda_start=torch.cuda.Event(True); cuda_end=torch.cuda.Event(True); cuda_start.record(); image_scores,pixel_scores,labels,pixel_labels=scores(test_loader,extractor,stats,config,True); cuda_end.record(); torch.cuda.synchronize(device); full=time.perf_counter()-start
 pred=image_scores>image_threshold; normal=labels==0; defective=labels==1; result={'methodology_id':METHODOLOGY_ID,'category':'bottle','backbone':'resnet50','device':str(device),'pretrained_backbone_source':source,'image_size':config.image_size,'feature_layers':list(config.feature_layers),'score_weights':list(config.score_blend_weights),'seed':a.seed,'fit_images':len(fitting_indices),'calibration_images':len(calibration_indices),'test_images':len(test),'calibration':'normal_only empirical quantile, image target FPR=0.02, pixel quantile=0.995','image_threshold':image_threshold,'pixel_threshold':pixel_threshold,'image_roc_auc':float(roc_auc_score(labels,image_scores)),'pixel_roc_auc':float(roc_auc_score(pixel_labels,pixel_scores)),'image_average_precision':float(average_precision_score(labels,image_scores)),'pixel_average_precision':float(average_precision_score(pixel_labels,pixel_scores)),'fpr':float(pred[normal].mean()),'fnr':float((~pred[defective]).mean()),'full_fit_calibration_evaluation_seconds':full,'test_model_only_cuda_ms':float(cuda_start.elapsed_time(cuda_end)),'test_model_only_cuda_ms_per_image':float(cuda_start.elapsed_time(cuda_end)/len(test)),'peak_cuda_memory_bytes':int(torch.cuda.max_memory_allocated(device)),'legacy_checkpoint':str(legacy),'legacy_checkpoint_sha256':digest(legacy),'threshold_provenance':'normal-only calibration split; final test labels and masks excluded'}
 a.output.mkdir(parents=True); (a.output/'split_manifest.json').write_text(json.dumps({'seed':a.seed,'fit_indices':fitting_indices,'calibration_indices':calibration_indices,'fit_paths':[str(train.image_paths[i]) for i in fitting_indices],'calibration_paths':[str(train.image_paths[i]) for i in calibration_indices]},indent=2)); torch.save({'methodology_id':METHODOLOGY_ID,'statistics':stats,'config':config.to_dict(),'image_threshold':image_threshold,'pixel_threshold':pixel_threshold,'provenance':result},a.output/'model.pt'); (a.output/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
