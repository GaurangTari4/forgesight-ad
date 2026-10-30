"""PatchCore-style frozen-feature anomaly detection with bounded NN search.

This implementation uses normalized, spatially aligned ResNet layer embeddings;
the coreset is exact greedy k-center over a deterministic candidate subset.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import uuid
from typing import Any
import torch
import torch.nn.functional as F

METHODOLOGY_ID="patchcore_v1_approx_greedy_coreset"

class PatchCoreArtifactError(ValueError): pass

def patch_embeddings(features:dict[str,torch.Tensor], layers:tuple[str,...])->tuple[torch.Tensor,tuple[int,int]]:
    target=features[layers[0]].shape[-2:]; aligned=[]
    for layer in layers:
        x=features[layer]
        if x.shape[-2:]!=target:x=F.interpolate(x,size=target,mode='bilinear',align_corners=False)
        aligned.append(x)
    merged=F.normalize(torch.cat(aligned,dim=1),p=2,dim=1)
    return merged.permute(0,2,3,1).reshape(-1,merged.shape[1]),target

def greedy_coreset(candidates:torch.Tensor,size:int,seed:int=42)->torch.Tensor:
    """Deterministic greedy k-center selection, not random sampling."""
    if not 0<size<=len(candidates):raise ValueError('invalid coreset size')
    gen=torch.Generator(device='cpu');gen.manual_seed(seed); first=int(torch.randint(len(candidates),(1,),generator=gen))
    chosen=[first]; nearest=torch.cdist(candidates,candidates[first:first+1]).squeeze(1)
    for _ in range(1,size):
        index=int(torch.argmax(nearest)); chosen.append(index)
        nearest=torch.minimum(nearest,torch.cdist(candidates,candidates[index:index+1]).squeeze(1))
    return torch.tensor(chosen,dtype=torch.long)

def nearest_distances(query:torch.Tensor,memory:torch.Tensor,query_chunk:int=1024,memory_chunk:int=4096)->torch.Tensor:
    if query_chunk<=0 or memory_chunk<=0:raise ValueError('chunk sizes must be positive')
    result=[]
    for q in query.split(query_chunk):
        best=torch.full((len(q),),float('inf'),device=q.device)
        for m in memory.split(memory_chunk):best=torch.minimum(best,torch.cdist(q,m).min(1).values)
        result.append(best)
    return torch.cat(result)

@torch.inference_mode()
def anomaly_map(extractor,images:torch.Tensor,memory:torch.Tensor,layers:tuple[str,...],query_chunk:int=1024,memory_chunk:int=4096)->torch.Tensor:
    from .autoencoder_pipeline import imagenet_normalize
    patches,shape=patch_embeddings(extractor(imagenet_normalize(images)),layers);dist=nearest_distances(patches,memory.to(images.device),query_chunk,memory_chunk)
    low=dist.reshape(images.shape[0],1,*shape);return F.interpolate(low,size=images.shape[-2:],mode='bilinear',align_corners=False)

def validate_artifact(payload:dict[str,Any],backbone:str|None=None)->None:
    required={'methodology_id','backbone_name','feature_layers','image_size','memory_bank','coreset','seed','split_manifest_sha256'}
    if not isinstance(payload,dict) or payload.get('methodology_id')!=METHODOLOGY_ID or required-payload.keys():raise PatchCoreArtifactError('Malformed or incompatible PatchCore artifact')
    if backbone and payload['backbone_name']!=backbone:raise PatchCoreArtifactError('Selected backbone does not match artifact')
    bank=payload['memory_bank']
    if not isinstance(bank,torch.Tensor) or bank.ndim!=2 or not torch.isfinite(bank).all():raise PatchCoreArtifactError('Invalid PatchCore memory bank')

def save_artifact(path:Path,payload:dict[str,Any])->None:
    """Atomically publish a new trusted artifact without replacing an existing one."""
    validate_artifact(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing PatchCore artifact: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        # Prefer a hard link: it is atomic no-replace publication on the same
        # volume. Some managed Windows volumes forbid links, so fall back to an
        # exclusive empty placeholder then replace only that placeholder.
        try:
            os.link(temporary, path)
        except OSError:
            with path.open("xb"):
                pass
            try:
                os.replace(temporary, path)
            except Exception:
                if path.exists() and path.stat().st_size == 0:
                    path.unlink()
                raise
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    else:
        if temporary.exists():
            temporary.unlink()

def load_artifact(path:Path,device:torch.device,backbone:str|None=None)->dict[str,Any]:
    try: payload=torch.load(path,map_location=device,weights_only=True)
    except Exception as exc:raise PatchCoreArtifactError('Could not safely load PatchCore artifact') from exc
    validate_artifact(payload,backbone);return payload
