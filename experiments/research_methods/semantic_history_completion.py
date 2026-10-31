"""Train-only semantic history completion with the main encoder's tied decoder."""
import numpy as np
import time
import torch
from torch.nn import functional as F
from research_methods.joint_semantic_ssc import ResearchModel as JointModel

SOURCE_FILES=['joint_semantic_ssc.py','../semantic_history_completion_protocol.md']


def completion_loss(projection, semantics, targets, visible, hidden, temperature=.2):
    # Ragged sets are small (4..40 items in released LastFM), packed without truncation.
    ids=torch.tensor([i for row in visible for i in row],device=semantics.device,dtype=torch.long)
    lengths=torch.tensor([len(row) for row in visible],device=semantics.device)
    owner=torch.repeat_interleave(torch.arange(len(visible),device=semantics.device),lengths)
    z=F.elu(projection(semantics[ids]))
    pooled=z.new_zeros((len(visible),z.shape[1])).index_add(0,owner,z)/lengths[:,None]
    query=F.normalize(pooled@projection.weight,dim=1,eps=1e-8)
    logits=query@targets.T/temperature
    mask=torch.zeros_like(logits,dtype=torch.bool)
    mask[owner,ids]=True
    logp=F.log_softmax(logits.masked_fill(mask,float('-inf')),dim=1)
    loss=torch.stack([-logp[k,torch.tensor(row,device=semantics.device)].mean()
                      for k,row in enumerate(hidden)]).mean()
    return loss,query


class ResearchModel(JointModel):
    def __init__(self,config,dataset,*args):
        super().__init__(config,dataset,*args)
        r=dataset.UserItemNet.tocsr()
        self.histories=[np.sort(r.indices[r.indptr[u]:r.indptr[u+1]]).copy() for u in range(r.shape[0])]
        self.eligible=np.array([u for u,h in enumerate(self.histories) if len(h)>=2])
        assert len(self.eligible)>0
        normalized=F.normalize(self.semantic_emb,dim=1)
        centered=normalized-normalized.mean(0,keepdim=True)
        assert bool((centered.norm(dim=1)>1e-8).all())
        self.register_buffer('completion_targets',F.normalize(centered,dim=1).detach())
        self.aux_rng=np.random.RandomState(2020201)
        self.aux_weight=.01
        self.aux_timing=[]
        self.method_config.update(name='semantic_history_completion',parent_commit='393d7cbc5d41fdc76e26ede2e2574fd3c85cf102',
            extra_loss='train-only masked-history multi-positive CE, semantic_map encoder and transpose tied decoder',
            completion=dict(mask_fraction=.2,temperature=.2,weight=.01,users_per_step=64,
                sampling_seed=2020201,targets='fixed unit semantics minus catalog mean, renormalized',
                inference='unchanged joint SSC',private_predictor=False,extra_parameters=0))

    def get_extra_state(self):return {'aux_rng':self.aux_rng.get_state()}

    def set_extra_state(self,state):self.aux_rng.set_state(state['aux_rng'])

    def sample_completion(self):
        selected=self.aux_rng.choice(self.eligible,size=min(64,len(self.eligible)),replace=False)
        visible=[];hidden=[]
        for u in selected:
            h=self.histories[u];count=min(len(h)-1,max(1,int(.2*len(h))))
            p=self.aux_rng.permutation(len(h));hidden.append(h[p[:count]].tolist());visible.append(h[p[count:]].tolist())
        return visible,hidden

    def history_loss(self):
        tick=time.perf_counter()
        visible,hidden=self.sample_completion()
        sampling=time.perf_counter()-tick
        events=None
        if self.semantic_emb.is_cuda:
            events=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
            events[0].record()
        loss=completion_loss(self.semantic_map,self.semantic_emb,self.completion_targets,visible,hidden)[0]
        if events:events[1].record()
        self.aux_timing.append((sampling,events))
        return loss

    def bpr_loss(self,users,pos,neg):
        loss,reg=super().bpr_loss(users,pos,neg)
        if self.training and self.aux_weight:
            aux=self.history_loss()
            self.latest_components.update(history_completion=float(aux.detach()),weighted_history_completion=float(self.aux_weight*aux.detach()))
            loss=loss+self.aux_weight*aux
        return loss,reg

    def evaluation_diagnostics(self):
        values=super().evaluation_diagnostics()
        values.update(semantic_map_weight_norm=float(self.semantic_map.weight.norm()),
                      semantic_map_bias_norm=float(self.semantic_map.bias.norm()))
        return values
