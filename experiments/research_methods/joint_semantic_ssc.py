"""Joint train-interaction/LLM-neighbor graph with spectrum shift correction.

Independent adaptation of SSC (KDD 2025), not an author-code reproduction.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import torch
from torch.nn import functional as F
from model import CoLaKG

SOURCE_FILES=['../joint_semantic_ssc_protocol.md',
              '../research_records/joint_semantic_ssc_sources_20261031.json']


def build_operator(interactions, neighbors):
    """All structural inputs are train interactions and fixed semantic top-K."""
    r=interactions.astype(np.float64).tocsr(copy=True)
    r.data[:]=1.;r.eliminate_zeros()
    nu,ni=r.shape
    neighbors=np.asarray(neighbors,dtype=np.int64)
    assert neighbors.shape[0]==ni and neighbors.min()>=0 and neighbors.max()<ni
    rows=np.repeat(np.arange(ni),neighbors.shape[1])
    s=sp.csr_matrix((np.ones(neighbors.size),(rows,neighbors.ravel())),shape=(ni,ni))
    s.setdiag(0);s.eliminate_zeros();s=s.maximum(s.T);s.data[:]=1.
    augmented=sp.bmat([[None,r],[r.T,s]],format='csr')
    degrees=np.asarray(augmented.sum(1)).ravel()
    inv=np.zeros_like(degrees);inv[degrees>0]=degrees[degrees>0]**-.5
    normalized=(sp.diags(inv)@augmented@sp.diags(inv)).tocsr()
    assert (normalized-normalized.T).nnz==0 or abs(normalized-normalized.T).max()<1e-12
    initial=np.random.RandomState(2020).standard_normal(nu+ni)
    value,vector=eigsh(normalized,k=1,which='SA',tol=1e-9,v0=initial)
    low=float(value[0]);residual=float(np.linalg.norm(normalized@vector[:,0]-low*vector[:,0]))
    assert -1.000001<=low<1. and residual<1e-6,(low,residual)
    # Symmetric nonnegative normalized adjacency has maximum eigenvalue 1.
    upper=1.;mu=(upper+low)/2.;delta=(upper-low)/2.
    assert delta>1e-6
    corrected=((normalized-mu*sp.eye(nu+ni,format='csr'))/delta).tocsr()
    meta=dict(users=nu,items=ni,train_edges=int(r.nnz),semantic_directed_entries=int(s.nnz),
        joint_entries=int(normalized.nnz),joint_min_eigenvalue=low,joint_max_eigenvalue=upper,
        eigenpair_residual=residual,shift_mu=mu,shift_delta=delta,
        correction_min_eigenvalue=(low-mu)/delta,correction_max_eigenvalue=(upper-mu)/delta,
        graph_source='binary train UI + symmetric union of original fixed top10 II; each raw edge weight 1',
        graph_estimation_labels='train only, no test labels or fitted recommendation checkpoints')
    return corrected,meta


class ResearchModel(CoLaKG):
    def __init__(self, config, dataset, *args):
        super().__init__(config,dataset,*args)
        assert not self.A_split and not config['use_drop_edge']
        operator,meta=build_operator(dataset.UserItemNet,self.adj_matrix.detach().cpu().numpy())
        coo=operator.tocoo()
        indices=torch.tensor(np.stack([coo.row,coo.col]),dtype=torch.long)
        values=torch.tensor(coo.data,dtype=torch.float32)
        self.register_buffer('joint_operator',torch.sparse_coo_tensor(indices,values,coo.shape).coalesce())
        # This replaces the old pre-UI attention enhancement, not stacks it twice.
        for name in ['W','a','W_u','a_u']:delattr(self,name)
        self.latest_components={}
        self.method_config=dict(name='joint_semantic_ssc',A_enabled=False,
            encoder='original ID + user/item semantic Linear/ELU/dropout mean inputs; joint UI/II spectral propagation',
            replaced='original one-hop item attention enhancement and pure UI propagation',
            effective_dropout=dict(item=config['dropout_i'],user=config['dropout_u'],neighbor_attention='inactive: attention module replaced by fixed symmetric graph'),
            layer_readout='0..3 corrected joint graph powers, arithmetic mean',
            graph=meta,extra_loss=False,source='SSC KDD2025; automatic exact endpoint correction, one actual LLM semantic source',
            differences_from_author='retain CoLaKG user/item semantic maps and original six-vector regularizer/initialization; use original non-PCA top10; original sigmoid-dot eval; no freerec/PyG',
            limitation='exact range correction does not prove optimal filter or preserve all informative frequencies')

    def computer(self):
        def merged(ids,semantics,projection,dropout):
            value=F.dropout(semantics,dropout,training=self.training)
            value=F.elu(projection(value))
            return .5*(ids+F.dropout(value,dropout,training=self.training))
        i=merged(self.embedding_item.weight,self.semantic_emb,self.semantic_map,self.dropout_i)
        u=merged(self.embedding_user.weight,self.user_semantic_emb,self.user_semantic_map,self.dropout_u)
        current=torch.cat([u,i],0);layers=[current]
        for _ in range(self.n_layers):
            current=torch.sparse.mm(self.joint_operator,current);layers.append(current)
        return torch.split(torch.stack(layers,0).mean(0),[self.num_users,self.num_items])

    def bpr_loss(self,users,pos,neg):
        loss,reg=super().bpr_loss(users,pos,neg)
        self.latest_components=dict(bpr=float(loss.detach()),six_vector_regularizer=float(reg.detach()))
        return loss,reg

    def begin_evaluation(self):self._eval_embeddings=self.computer()

    def end_evaluation(self):
        if hasattr(self,'_eval_embeddings'):del self._eval_embeddings

    def getUsersRating(self,users):
        u,i=self._eval_embeddings if hasattr(self,'_eval_embeddings') else self.computer()
        return self.f(u[users.long()]@i.T)

    def evaluation_diagnostics(self):
        u,i=self._eval_embeddings
        return dict(**self.latest_components,item_norm_mean=float(i.norm(dim=1).mean()),
            user_norm_mean=float(u.norm(dim=1).mean()),
            joint_graph_shift_mu=self.method_config['graph']['shift_mu'],
            joint_graph_shift_delta=self.method_config['graph']['shift_delta'])
