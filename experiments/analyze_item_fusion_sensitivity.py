"""Local sensitivity around 1:1 on frozen models; no fitting or test ranking."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from pathlib import Path
import copy,csv,json,hashlib,uuid
from datetime import datetime
import numpy as np
import scipy.sparse as sp
import torch
from torch.nn import functional as F
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from check_dynamic_semantic_attention import ROOT
from model import CoLaKG
from research_methods.asymmetric_semantic_attention import ResearchModel as Asymmetric
from research_methods.content_neighbor_fusion import ResearchModel as Fusion
torch.set_num_threads(2)
RUN=Path(r'D:\CoLaKGResearch\content-neighbor-fusion-v1\direct-test-content_neighbor_fusion-s2020-20261030-025447-df2704')
BASE=Path(r'D:\CoLaKGExperiments\lastfm-baseline-20261022-233837-33d715d8')
PARENT=Path(r'D:\CoLaKGResearch\asymmetric-semantic-attention-v1\direct-test-asymmetric_semantic_attention-s2020-20261028-233455-8d6c5a')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class Probe:
    computer=Fusion.computer
    def fuse_messages(self,x,m):
        self.center,self.message=x.detach(),m.detach()
        return x+self.probe_gate[:,None]*(m-x)
class OriginalProbe(Probe,CoLaKG):
    def pair_logits(self,neighbors,centers):
        k=neighbors@self.W;q=(centers@self.W)[:,None,:].expand_as(k)
        return F.leaky_relu((torch.cat([k,q],-1)@self.a).squeeze(-1),.2)
class ParentProbe(Probe,Asymmetric):pass

def main():
    out=Path(r'D:\CoLaKGExperiments')/('item-fusion-sensitivity-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]);out.mkdir()
    manifest=json.loads((RUN/'manifest.json').read_text());config=manifest['config'];data=ROOT/'data/lastfm'
    inputs={}
    for name in ['train.txt','lastfm_embeddings_simcse_kg.pt','lastfm_embeddings_simcse_kg_user.pt']:
        p=data/name;inputs[str(p)]=sha(p);assert inputs[str(p)]==manifest['input_sha256'][str(p)]
    item=torch.load(data/'lastfm_embeddings_simcse_kg.pt',weights_only=True,map_location='cpu')
    user=torch.load(data/'lastfm_embeddings_simcse_kg_user.pt',weights_only=True,map_location='cpu')
    knn=torch.tensor(np.argsort(-cosine_similarity(item.numpy()),axis=1)[:,1:11]).long()
    a=sp.load_npz(RUN/'graph-cache.npz').tocoo()
    graph=torch.sparse_coo_tensor(torch.tensor(np.stack([a.row,a.col])),torch.tensor(a.data,dtype=torch.float32),a.shape).coalesce()
    class Data:
        n_users,m_items=len(user),len(item)
        def getSparseGraph(self):return graph
    positives={};pairs=[]
    for line in (data/'train.txt').read_text().splitlines():
        ids=list(map(int,line.split()))
        if ids:positives.setdefault(ids[0],set()).update(ids[1:]);pairs.extend((ids[0],i) for i in ids[1:])
    u,p=torch.tensor(pairs).T
    weights=torch.tensor([1/len(positives[int(uid)]) for uid in u],dtype=torch.float32);weights/=weights.sum()
    degree=np.bincount(p.numpy(),minlength=len(item));negative_sets=[]
    for seed in [410,411,412]:
        rng=np.random.RandomState(seed);neg=[]
        for uid in u.tolist():
            j=rng.randint(len(item))
            while j in positives[uid]:j=rng.randint(len(item))
            neg.append(j)
        negative_sets.append(torch.tensor(neg))
    summary=dict(inputs=inputs,train_positive_count=len(pairs),test_ranking_computed=False,optimizer_steps=0,
        objective='uniform active train users, all their positives averaged, one uniform unobserved negative per positive in each of 3 draws',
        meaning='Derivative of total training BPR wrt one item pre-propagation scalar neighbor ratio at .5; includes effects on other nodes. Not optimal ratio or generalization evidence.',models={})
    refs=[('original_final2000',OriginalProbe,BASE/'checkpoints/colakg-lastfm-latest.pth.tar',json.loads((BASE/'result.json').read_text())['final_model_sha256']),
          ('asymmetric_best1650',ParentProbe,PARENT/'checkpoints/best-protocol.pt',json.loads((PARENT/'result.json').read_text())['best_observed_original_protocol']['checkpoint_sha256'])]
    for label,cls,path,expected in refs:
        assert sha(path)==expected
        state=torch.load(path,map_location='cpu',weights_only=False);state=state.get('model',state)
        model=cls(config,Data(),knn,item,user).eval();model.load_state_dict(state)
        for param in model.parameters():param.requires_grad_(False)
        frozen=copy.deepcopy(model.state_dict());grads=[];losses=[]
        def loss(g,n):
            model.probe_gate=g;us,it=model.computer()
            return (F.softplus((us[u]*(it[n]-it[p])).sum(1))*weights).sum()
        # Check numerical bridge against unmodified full original / parent model.
        reference=(CoLaKG if label.startswith('original') else Asymmetric)(config,Data(),knn,item,user).eval()
        reference.load_state_dict(state)
        with torch.no_grad():
            model.probe_gate=torch.full((len(item),),.5)
            for x,y in zip(model.computer(),reference.computer()):torch.testing.assert_close(x,y)
        del reference
        for neg in negative_sets:
            g=torch.full((len(item),),.5,requires_grad=True);l=loss(g,neg)
            grads.append(torch.autograd.grad(l,g)[0].numpy());losses.append(float(l))
        grads=np.stack(grads);mean=grads.mean(0)
        classes=np.where((grads < -1e-9).all(0),'more_neighbor',np.where((grads > 1e-9).all(0),'more_center','uncertain'))
        with torch.no_grad():
            sweep={str(v):float(loss(torch.full((len(item),),v),negative_sets[0])) for v in [0.,.25,.45,.5,.55,.75,1.]}
            fd=[]
            for i in [0,len(item)//2,len(item)-1]:
                g=torch.full((len(item),),.5);g[i]+=.01;plus=float(loss(g,negative_sets[0]))
                g[i]-=.02;minus=float(loss(g,negative_sets[0]));observed=(plus-minus)/.02
                assert np.isclose(observed,grads[0,i],rtol=.04,atol=2e-7),(i,observed,grads[0,i])
                fd.append(dict(item_id=i,finite_difference=observed,autograd=float(grads[0,i])))
            model.probe_gate=torch.full((len(item),),.5);model.computer()
        cosine=F.cosine_similarity(model.center,model.message).numpy();ratio=(model.message.norm(dim=1)/model.center.norm(dim=1).clamp_min(1e-8)).numpy()
        thresholds=np.quantile(degree,[1/3,2/3]);groups=np.digitize(degree,thresholds,right=True)
        rows=[]
        for name,feature in [('train_degree',degree),('center_message_cosine',cosine),('message_center_norm_ratio',ratio)]:
            edges=np.quantile(feature,[1/3,2/3]);idx=np.digitize(feature,edges,right=True)
            for group in range(3):
                take=idx==group;rows.append(dict(feature=name,group=group,n=int(take.sum()),min=float(feature[take].min()),max=float(feature[take].max()),
                    counts={c:int(((classes==c)&take).sum()) for c in ['more_neighbor','more_center','uncertain']}))
        with (out/(label+'-items.csv')).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.writer(f);w.writerow(['item_id','train_degree','center_message_cosine','message_center_norm_ratio','local_direction','gradient_mean','gradient_410','gradient_411','gradient_412'])
            for i in range(len(item)):w.writerow([i,int(degree[i]),float(cosine[i]),float(ratio[i]),classes[i],float(mean[i]),*map(float,grads[:,i])])
        s=dict(checkpoint=str(path),sha256=expected,ratio_half_train_bpr=losses,global_ratio_sweep_training_only=sweep,
            counts={c:int((classes==c).sum()) for c in ['more_neighbor','more_center','uncertain']},groups=rows,finite_difference_checks=fd,
            spearman_descriptive={name:float(spearmanr(feature,-mean).statistic) for name,feature in [('degree',degree),('cosine',cosine),('norm_ratio',ratio)]})
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in frozen.items());assert sha(path)==expected
        summary['models'][label]=s
        print(label,json.dumps(s,ensure_ascii=False),flush=True)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print('OUTPUT='+str(out),flush=True)
if __name__=='__main__':main()
