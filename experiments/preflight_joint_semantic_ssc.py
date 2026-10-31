"""One CPU real-input forward/backward check; no fitting, evaluation or weights."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import hashlib,json,traceback,uuid
from datetime import datetime
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics.pairwise import cosine_similarity
from check_dynamic_semantic_attention import ROOT
from research_methods.joint_semantic_ssc import ResearchModel

torch.set_num_threads(2)
def main():
    out=Path(r'D:\CoLaKGExperiments')/('joint-ssc-preflight-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]);out.mkdir()
    try:
        old=Path(r'D:\CoLaKGResearch\top20-catalog-ranking-v1\direct-test-top20_catalog_ranking-s2020-20261031-081145-9d92f1')
        manifest=json.loads((old/'manifest.json').read_text(encoding='utf8'));data=ROOT/'data/lastfm'
        hashes={}
        for name in ['train.txt','lastfm_embeddings_simcse_kg.pt','lastfm_embeddings_simcse_kg_user.pt']:
            p=data/name;hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
            assert hashes[str(p)]==manifest['input_sha256'][str(p)]
        item=torch.load(data/'lastfm_embeddings_simcse_kg.pt',map_location='cpu',weights_only=True)
        user=torch.load(data/'lastfm_embeddings_simcse_kg_user.pt',map_location='cpu',weights_only=True)
        positives={}
        for line in (data/'train.txt').read_text().splitlines():
            ids=list(map(int,line.split()))
            if ids:positives[ids[0]]=ids[1:]
        pairs=[(u,i) for u,items in positives.items() for i in items];rows,cols=np.asarray(pairs).T
        r=sp.csr_matrix((np.ones(len(pairs)),(rows,cols)),shape=(len(user),len(item)))
        a=sp.bmat([[None,r],[r.T,None]],format='coo');deg=np.asarray(a.sum(1)).ravel();iv=deg**-.5
        a=(sp.diags(iv)@a@sp.diags(iv)).tocoo()
        graph=torch.sparse_coo_tensor(torch.tensor(np.stack([a.row,a.col])),torch.tensor(a.data,dtype=torch.float32),a.shape).coalesce()
        class Data:
            n_users,m_items=len(user),len(item)
            UserItemNet=r
            def getSparseGraph(self):return graph
        neighbors=torch.tensor(np.argsort(-cosine_similarity(item.numpy()),axis=1)[:,1:11]).long()
        torch.manual_seed(2020)
        model=ResearchModel(manifest['config'],Data(),neighbors,item,user)
        rng=np.random.RandomState(2020);sample=[]
        for u in rng.choice(sorted(positives),size=1024):
            p=rng.choice(positives[u]);n=rng.randint(len(item))
            while n in positives[u]:n=rng.randint(len(item))
            sample.append((u,p,n))
        u,p,n=torch.tensor(sample).T
        model.train();torch.manual_seed(77);loss,reg=model.bpr_loss(u,p,n)
        (loss+1e-4*reg).backward()
        assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
        gradients={k:float(v.grad.norm()) for k,v in model.named_parameters() if v.grad is not None}
        model.eval()
        with torch.no_grad():
            evloss,_=model.bpr_loss(u,p,n);ue,ie=model.computer()
        report=dict(success=True,optimizer_steps=0,test_ranking_computed=False,input_sha256=hashes,
            method_config=model.method_config,train_bpr=float(loss),eval_bpr=float(evloss),
            weighted_regularizer=float(1e-4*reg),gradient_norms=gradients,
            eval_item_norm_mean=float(ie.norm(dim=1).mean()),parameters=sum(v.numel() for v in model.parameters()))
        (out/'preflight.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        print(json.dumps(report,ensure_ascii=False),flush=True);print('OUTPUT='+str(out),flush=True)
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf8');raise
if __name__=='__main__':main()
