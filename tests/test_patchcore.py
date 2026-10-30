import pytest
torch=pytest.importorskip('torch')
from forgesight_ad.patchcore_pipeline import greedy_coreset,nearest_distances,patch_embeddings,validate_artifact,PatchCoreArtifactError,METHODOLOGY_ID,save_artifact,load_artifact

def test_embeddings_align_and_normalize():
 p,shape=patch_embeddings({'a':torch.ones(1,2,4,4),'b':torch.ones(1,3,2,2)},('a','b'));assert shape==(4,4) and p.shape==(16,5);assert torch.allclose(p.norm(dim=1),torch.ones(16))
def test_greedy_is_deterministic_and_selects_extremes():
 x=torch.tensor([[0.],[1.],[10.],[11.]]);a=greedy_coreset(x,2,1);assert torch.equal(a,greedy_coreset(x,2,1));assert len(set(a.tolist()))==2
def test_chunked_nn_matches_reference():
 q=torch.tensor([[0.,0.],[2.,0.]]);m=torch.tensor([[0.,0.],[3.,0.]]);assert torch.allclose(nearest_distances(q,m,1,1),torch.tensor([0.,1.]))
def test_rejects_malformed_artifact():
 with pytest.raises(PatchCoreArtifactError):validate_artifact({})

def _payload():
 return {'methodology_id':METHODOLOGY_ID,'category':'cable','backbone_name':'resnet18','feature_layers':['layer2','layer3'],'image_size':512,'memory_bank':torch.ones(2,3),'coreset':{},'seed':42,'split_manifest_sha256':'x'}

def test_atomic_save_allows_existing_parent_and_rejects_overwrite(tmp_path):
 path=tmp_path/'existing_parent'/'model.pt';path.parent.mkdir();save_artifact(path,_payload());assert load_artifact(path,torch.device('cpu'),'resnet18')['category']=='cable'
 with pytest.raises(FileExistsError):save_artifact(path,_payload())

def test_failed_save_cleans_temporary_file(tmp_path,monkeypatch):
 import forgesight_ad.patchcore_pipeline as pipeline
 path=tmp_path/'new_parent'/'model.pt';monkeypatch.setattr(pipeline.torch,'save',lambda *_:(_ for _ in ()).throw(RuntimeError('interrupted')))
 with pytest.raises(RuntimeError,match='interrupted'):save_artifact(path,_payload())
 assert not path.exists() and not list(path.parent.glob('*.tmp'))
