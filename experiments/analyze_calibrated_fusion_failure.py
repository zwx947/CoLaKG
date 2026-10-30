"""Frozen CPU diagnostics; no optimizer, test ranking or fitting."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import copy,hashlib,json,types,uuid
from datetime import datetime
import numpy as np
import scipy.sparse as sp
import torch
from torch.nn import functional as F
from sklearn.metrics.pairwise import cosine_similarity
from check_dynamic_semantic_attention import ROOT
from research_methods.calibrated_neighbor_fusion import ResearchModel

torch.set_num_threads(2)
RUN=Path(r'D:\CoLaKGResearch\calibrated-neighbor-fusion-v1\direct-test-calibrated_neighbor_fusion-s2020-20261030-175809-4afb9d')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def geometry(v):
    v=v.detach();u=F.normalize(v,dim=1);n=len(v)
    c=v-v.mean(0);eig=torch.linalg.eigvalsh(c.T@c).clamp_min(0);q=eig/eig.sum().clamp_min(1e-12)
    return dict(norm_mean=float(v.norm(dim=1).mean()),
        mean_off_diagonal_cosine=float((u.sum(0).square().sum()-n)/(n*(n-1))),
        effective_covariance_rank=float(torch.exp(-(q*q.clamp_min(1e-30).log()).sum())),
        centered_energy_fraction=float(c.square().sum()/v.square().sum().clamp_min(1e-12)))
def cos(a,b):return float(F.cosine_similarity(a,b).mean())

def main():
    out=Path(r'D:\CoLaKGExperiments')/('calibrated-fusion-analysis-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]);out.mkdir()
    manifest=json.loads((RUN/'manifest.json').read_text());result=json.loads((RUN/'result.json').read_text())
    data=ROOT/'data/lastfm'
    for name in ['train.txt','lastfm_embeddings_simcse_kg.pt','lastfm_embeddings_simcse_kg_user.pt']:
        assert sha(data/name)==manifest['input_sha256'][str(data/name)]
    for rel,h in manifest['source_sha256'].items():assert sha(ROOT/rel)==h,rel
    item=torch.load(data/'lastfm_embeddings_simcse_kg.pt',map_location='cpu',weights_only=True)
    user=torch.load(data/'lastfm_embeddings_simcse_kg_user.pt',map_location='cpu',weights_only=True)
    neighbors=torch.tensor(np.argsort(-cosine_similarity(item.numpy()),axis=1)[:,1:11]).long()
    a=sp.load_npz(RUN/'graph-cache.npz').tocoo();graph=torch.sparse_coo_tensor(torch.tensor(np.stack([a.row,a.col])),torch.tensor(a.data,dtype=torch.float32),a.shape).coalesce()
    positive={}
    for line in (data/'train.txt').read_text().splitlines():
        ids=list(map(int,line.split()))
        if ids:positive.setdefault(ids[0],set()).update(ids[1:])
    pairs=[(u,i) for u,items in positive.items() for i in sorted(items)]
    uu,ii=np.array(pairs).T
    net=sp.csr_matrix((np.ones(len(pairs)),(uu,ii)),shape=(len(user),len(item)))
    class Data:
        n_users,m_items=len(user),len(item)
        UserItemNet=net
        def getSparseGraph(self):return graph
    rng=np.random.RandomState(3010425);active=np.array(sorted(positive));sample=[]
    for _ in range(4096):
        u=int(rng.choice(active));p=int(rng.choice(sorted(positive[u])));n=rng.randint(len(item))
        while n in positive[u]:n=rng.randint(len(item))
        sample.append((u,p,n))
    users,pos,neg=torch.tensor(sample).T
    output=dict(run=str(RUN),no_optimizer=True,test_ranking_computed=False,
        input_source_hashes_verified=True,sample_sha256=hashlib.sha256(np.asarray(sample,dtype=np.int64).tobytes()).hexdigest(),checkpoints={})
    for name,expected in [('best-protocol.pt',result['best_observed_original_protocol']['checkpoint_sha256']),('final.pt',result['final_checkpoint_sha256'])]:
        path=RUN/'checkpoints'/name;assert sha(path)==expected
        state=torch.load(path,map_location='cpu',weights_only=False)
        model=ResearchModel(manifest['config'],Data(),neighbors,item,user).eval();model.load_state_dict(state['model'])
        before=copy.deepcopy(model.state_dict());capture={};original=model.fuse_messages
        def wrapped(self,x,m):
            z=original(x,m);capture.update(x=x.detach(),m=m.detach(),z=z.detach());return z
        model.fuse_messages=types.MethodType(wrapped,model)
        handle=model.message_adapter.register_forward_hook(lambda mod,args,value:capture.update(t=value.detach()))
        with torch.no_grad():
            u,it=model.computer();x,m,t,z=[capture[k].clone() for k in ['x','m','t','z']]
            g=model._fusion_gate.clone();margin=(u[users]*(it[pos]-it[neg])).sum(1)
            row=dict(epoch=state['epoch'],sha256=expected,
                train_probe_bpr=float(F.softplus(-margin).mean()),
                train_probe_margin_quantiles=torch.quantile(margin,torch.tensor([.01,.1,.5,.9,.99])).tolist(),
                geometry={k:geometry(v) for k,v in [('center',x),('message',m),('adapted',t),('fused',z),('propagated_items',it)]},
                cosines=dict(adapter_center=cos(t,x),adapter_original_message=cos(t,m),fused_center=cos(z,x),fused_adapter=cos(z,t)),
                gate_mean=float(g.mean()),degree_gate_spearman=None,
                trained_adapter_weight_norms={n:float(p.norm()) for n,p in model.message_adapter.named_parameters()})
            from scipy.stats import spearmanr
            row['degree_gate_spearman']=float(spearmanr(np.asarray(net.sum(0)).flatten(),g.flatten().numpy()).statistic)
            def adapt(a,b):return model.unit(model.message_adapter(model.features(model.unit(a),model.unit(b))))
            base=adapt(x,m);probes=[]
            for seed in [81,82,83]:
                torch.manual_seed(seed);dx=F.normalize(torch.randn_like(x),dim=1)*x.norm(dim=1,keepdim=True)*.01
                dm=F.normalize(torch.randn_like(m),dim=1)*m.norm(dim=1,keepdim=True)*.01
                perm=torch.randperm(len(x))
                probes.append(dict(seed=seed,
                    relative_001_center_perturbation_direction_change=float((adapt(x+dx,m)-base).norm(dim=1).mean()),
                    relative_001_message_perturbation_direction_change=float((adapt(x,m+dm)-base).norm(dim=1).mean()),
                    shuffled_center_adapter_cosine=cos(base,adapt(x[perm],m)),
                    shuffled_message_adapter_cosine=cos(base,adapt(x,m[perm]))))
            row['adapter_input_sensitivity']=probes
            noise=[]
            for mode in ['full','input_only','neighbor_only']:
                model.dropout_i=.6 if mode in ['full','input_only'] else 0.
                model.dropout_u=.2 if mode in ['full','input_only'] else 0.
                model.dropout_neighbor=.6 if mode in ['full','neighbor_only'] else 0.
                for seed in [81,82,83]:
                    torch.manual_seed(seed);model.train();tu,ti=model.computer()
                    tmarg=(tu[users]*(ti[pos]-ti[neg])).sum(1)
                    noise.append(dict(mode=mode,seed=seed,train_probe_bpr=float(F.softplus(-tmarg).mean()),
                        margin_mean=float(tmarg.mean()),
                        gate_mean=float(model._fusion_gate.mean()),
                        fused_norm_mean=float(capture['z'].norm(dim=1).mean()),
                        propagated_item_norm_mean=float(ti.norm(dim=1).mean()),
                        fused_cosine_to_eval=cos(capture['z'],z),center_cosine_to_eval=cos(capture['x'],x),
                        propagated_item_cosine_to_eval=cos(ti,it)))
            row['training_dropout_forward_probes']=noise
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items());assert sha(path)==expected
        row['parameters_unchanged']=True;output['checkpoints'][name]=row;handle.remove()
        print(name,json.dumps(row,ensure_ascii=False),flush=True)
    # Same train-only samples on the preserved parent; different best epochs are
    # descriptive checkpoints, not a matched causal training comparison.
    from research_methods.asymmetric_semantic_attention import ResearchModel as Parent
    parent_run=Path(r'D:\CoLaKGResearch\asymmetric-semantic-attention-v1\direct-test-asymmetric_semantic_attention-s2020-20261028-233455-8d6c5a')
    pr=json.loads((parent_run/'result.json').read_text());output['parent_reference']={}
    for name,expected in [('best-protocol.pt',pr['best_observed_original_protocol']['checkpoint_sha256']),('final.pt',pr['final_checkpoint_sha256'])]:
        path=parent_run/'checkpoints'/name;assert sha(path)==expected
        state=torch.load(path,map_location='cpu',weights_only=False)
        model=Parent(manifest['config'],Data(),neighbors,item,user);model.load_state_dict(state['model'])
        before=copy.deepcopy(model.state_dict());rows=[]
        with torch.no_grad():
            for training in [False,True]:
                model.train(training)
                for seed in ([81,82,83] if training else [81]):
                    torch.manual_seed(seed);u,it=model.computer();margin=(u[users]*(it[pos]-it[neg])).sum(1)
                    rows.append(dict(training=training,seed=seed,bpr=float(F.softplus(-margin).mean())))
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items());assert sha(path)==expected
        output['parent_reference'][name]=dict(epoch=state['epoch'],sha256=expected,probes=rows)
    (out/'analysis.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print('OUTPUT='+str(out),flush=True)
if __name__=='__main__':main()
