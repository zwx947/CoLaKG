"""CPU-only frozen-checkpoint analysis. No optimizer, updates or test ranking."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '2'
import copy
import hashlib
import json
from pathlib import Path
import types
import uuid
from datetime import datetime
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics.pairwise import cosine_similarity
from check_dynamic_semantic_attention import ROOT
from research_methods.content_neighbor_fusion import ResearchModel

torch.set_num_threads(2)
RUN=Path(r'D:\CoLaKGResearch\content-neighbor-fusion-v1\direct-test-content_neighbor_fusion-s2020-20261030-025447-df2704')
OUT=Path(r'D:\CoLaKGExperiments')/('fusion-failure-analysis-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(t):
    t=t.detach().flatten().float()
    return dict(mean=float(t.mean()),std=float(t.std(unbiased=False)),min=float(t.min()),max=float(t.max()),
                quantiles=torch.quantile(t,torch.tensor([.01,.1,.5,.9,.99])).tolist())

def main():
    OUT.mkdir(exist_ok=False)
    manifest=json.loads((RUN/'manifest.json').read_text())
    result=json.loads((RUN/'result.json').read_text())
    evaluations=json.loads((RUN/'evaluations.json').read_text())
    history=json.loads((RUN/'history.json').read_text())
    inputs={};sources={}
    for rel,h in manifest['source_sha256'].items():
        if rel.replace('\\','/').startswith('rec_code/') or rel.replace('\\','/').startswith('experiments/research_methods/'):
            p=ROOT/rel;assert sha(p)==h,rel;sources[str(p)]=h
    data=ROOT/'data/lastfm'
    for name in ['train.txt','lastfm_embeddings_simcse_kg.pt','lastfm_embeddings_simcse_kg_user.pt']:
        p=data/name;h=sha(p);assert h==manifest['input_sha256'][str(p)];inputs[str(p)]=h
    item=torch.load(data/'lastfm_embeddings_simcse_kg.pt',map_location='cpu',weights_only=True)
    user=torch.load(data/'lastfm_embeddings_simcse_kg_user.pt',map_location='cpu',weights_only=True)
    neighbors=torch.tensor(np.argsort(-cosine_similarity(item.numpy()),axis=1)[:,1:11]).long()
    cached=sp.load_npz(RUN/'graph-cache.npz').tocoo()
    graph=torch.sparse_coo_tensor(torch.tensor(np.stack([cached.row,cached.col])),torch.tensor(cached.data,dtype=torch.float32),cached.shape).coalesce()
    class Data:
        n_users,m_items=len(user),len(item)
        def getSparseGraph(self):return graph
    positives={};pairs=[]
    for line in (data/'train.txt').read_text().splitlines():
        ids=list(map(int,line.split()))
        if not ids:continue
        positives.setdefault(ids[0],set()).update(ids[1:]);pairs.extend((ids[0],i) for i in ids[1:])
    rng=np.random.RandomState(30102026)
    selected=rng.choice(len(pairs),1024,replace=False)
    sample=[]
    for k in selected:
        u,i=pairs[k];j=rng.randint(len(item))
        while j in positives[u]:j=rng.randint(len(item))
        sample.append((u,i,j))
    users,pos,neg=torch.tensor(sample).T
    output={'run':str(RUN),'no_parameter_updates':True,'test_ranking_computed':False,
        'scope':'frozen CPU diagnostics; probes cannot identify historical early gradients or causal generalization effects',
        'sources':sources,'inputs':inputs,'sample_sha256':hashlib.sha256(np.asarray(sample,dtype=np.int64).tobytes()).hexdigest(),
        'timeline':[r for r in evaluations if r['completed_epochs'] in [0,5,10,20,50,100,200,500,1000,1300,1500,2000]],
        'loss_timeline':[r for r in history if r['completed_epochs'] in [1,5,20,100,500,1300,2000]],'checkpoints':{}}
    for name,expected in [('best-protocol.pt',result['best_observed_original_protocol']['checkpoint_sha256']),('final.pt',result['final_checkpoint_sha256'])]:
        path=RUN/'checkpoints'/name;assert sha(path)==expected
        # Trusted local checkpoint from this recorded run; loading is diagnostic only.
        state=torch.load(path,map_location='cpu',weights_only=False)
        model=ResearchModel(manifest['config'],Data(),neighbors,item,user)
        model.load_state_dict(state['model']);before=copy.deepcopy(model.state_dict())
        captured={};original=model.fuse_messages
        def capture(self,x,m):
            captured['center']=x;captured['message']=m
            return original(x,m)
        model.fuse_messages=types.MethodType(capture,model)
        hook=model.content_gate.register_forward_hook(lambda module,args,value:captured.update(logits=value))
        cases={}
        for mode in ['eval','train_full','train_inputs_only','train_neighbors_only']:
            model.train(mode!='eval')
            model.dropout_i=.6 if mode in ['eval','train_full','train_inputs_only'] else 0.
            model.dropout_u=.2 if mode in ['eval','train_full','train_inputs_only'] else 0.
            model.dropout_neighbor=.6 if mode in ['eval','train_full','train_neighbors_only'] else 0.
            rows=[]
            for seed in ([77] if mode=='eval' else [77,78,79]):
                torch.manual_seed(seed)
                with torch.no_grad():
                    model.computer();g=model._fusion_gate;x=captured['center'];m=captured['message'];z=captured['logits']
                    rows.append(dict(gate=stats(g),logits=stats(z),gate_below_001=float((g<.01).float().mean()),
                        gate_above_099=float((g>.99).float().mean()),exact_zero=float((g==0).float().mean()),exact_one=float((g==1).float().mean()),
                        sigmoid_derivative=stats(g*(1-g)),dimension_means=g.mean(0).tolist(),
                        variance_across_items=float(g.var(0,unbiased=False).mean()),
                        message_to_center_norm=stats(m.norm(dim=1)/x.norm(dim=1).clamp_min(1e-8)),
                        cosine=stats(torch.nn.functional.cosine_similarity(x,m)),
                        neighbor_term_norm=float((g*m).norm()),center_term_norm=float(((1-g)*x).norm())))
            cases[mode]=rows
        model.train();model.dropout_i=.6;model.dropout_u=.2;model.dropout_neighbor=.6;torch.manual_seed(77)
        loss,reg=model.bpr_loss(users,pos,neg)
        names,params=zip(*[(n,p) for n,p in model.named_parameters() if n.startswith('content_gate.')])
        bpr_grad=torch.autograd.grad(loss,params,retain_graph=True)
        logits_grad=torch.autograd.grad(loss,captured['logits'],retain_graph=True)[0]
        reg_grad=torch.autograd.grad(1e-4*reg,params)
        gradients={n:dict(bpr_norm=float(a.norm()),regularization_norm=float(b.norm())) for n,a,b in zip(names,bpr_grad,reg_grad)}
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
        assert sha(path)==expected
        output['checkpoints'][name]=dict(epoch=state['epoch'],sha256=expected,cases=cases,gradients=gradients,
            bpr=float(loss),weighted_regularizer=float(1e-4*reg),logits_gradient=stats(logits_grad),
            parameter_norms={n:float(p.norm()) for n,p in zip(names,params)},parameters_unchanged=True)
        hook.remove();del state,model,before
    (OUT/'analysis.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print('ANALYSIS_DIRECTORY='+str(OUT))
    for name,r in output['checkpoints'].items():
        ev=r['cases']['eval'][0]
        print(name,json.dumps({k:ev[k] for k in ['gate_below_001','gate_above_099','exact_zero','exact_one','variance_across_items','message_to_center_norm','neighbor_term_norm','center_term_norm']}))
        print('dimmeans',ev['dimension_means'])
        print('gradient',r['gradients'])
        for mode,rows in r['cases'].items(): print(mode,'gate',np.mean([x['gate']['mean'] for x in rows]),'normratio',np.mean([x['message_to_center_norm']['mean'] for x in rows]))

if __name__=='__main__':main()
