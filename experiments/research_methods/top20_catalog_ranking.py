"""Train-only, detached Top-20 weighting of a full-catalog ranking auxiliary.

Inspired by SL@K (KDD 2025); independent adaptation, not its reproduction.
"""
import torch
from torch.nn import functional as F
from model import CoLaKG

SOURCE_FILES=['../top20_catalog_ranking_protocol.md',
              '../research_records/top20_catalog_ranking_sources_20261030.json']


def catalog_terms(scores, positive_items, known_positives, k=20):
    """Known train positives are excluded from negatives, target added once.

    Weight and K-th negative threshold do not backpropagate; reducing a positive
    score cannot evade its own penalty by reducing its weight in the same step.
    """
    assert (known_positives.sum(1) <= scores.shape[1]-k).all()
    negative_scores=scores.masked_fill(known_positives, -torch.inf)
    target=scores.gather(1,positive_items[:,None]).squeeze(1)
    with torch.no_grad():
        beta=negative_scores.topk(k,dim=1).values[:,-1]
        weight=torch.sigmoid(target-beta)
    # Stable CE on target plus every not-known-positive item, raw dot scores.
    ce=F.softplus(torch.logsumexp(negative_scores,dim=1)-target)
    return (weight*ce).mean(),weight,beta,ce


class ResearchModel(CoLaKG):
    def __init__(self, config, dataset, *args):
        super().__init__(config,dataset,*args)
        self.register_buffer('train_positive_mask',torch.as_tensor(dataset.UserItemNet.toarray()>0))
        assert (self.train_positive_mask.sum(1)<=self.num_items-20).all()
        self.latest_components={}
        self.method_config=dict(name='top20_catalog_ranking',A_enabled=False,
            encoder='exact original CoLaKG; original attention, two 1:1 fusions, dropout and UI propagation',
            objective='original BPR + 0.05 * detached sigmoid(pos - kth unobserved score) * CE(target + all unobserved)',
            cutoff=20,auxiliary_weight=.05,score_temperature=1.,weight_temperature=1.,
            negative_mask='train known positives only; no test labels',
            score='unchanged raw dot product; inference unchanged original sigmoid(dot)',
            quantile='exact kth unobserved score from current training forward, detached',
            difference_from_SL_at_K='anchored BPR, exact masked catalog, detached weights, positive included in denominator, raw dot scores; not original bound or replication',
            parameter_delta=0,source='Breaking the Top-K Barrier, KDD 2025; independently implemented adaptation',
            regularization='unchanged original six sampled vectors')

    def bpr_loss(self, users, pos, neg):
        # Retains the original single forward and exact six-vector regularizer.
        all_users,all_items=self.computer()
        u,p,n=all_users[users],all_items[pos],all_items[neg]
        ru,rp,rn=self.embedding_user(users),self.embedding_item(pos),self.embedding_item(neg)
        mapped_items=self.semantic_map(self.semantic_emb)
        sp,sn=mapped_items[pos],mapped_items[neg]
        su=self.user_semantic_map(self.user_semantic_emb)[users]
        bpr=F.softplus((u*n).sum(1)-(u*p).sum(1)).mean()
        reg=sum(x.square().sum() for x in [ru,rp,rn,sp,sn,su])/(2*len(users))
        scores=u@all_items.t()
        catalog,weights,beta,ce=catalog_terms(scores,pos,self.train_positive_mask[users])
        self.latest_components=dict(bpr=float(bpr.detach()),catalog_top20=float(catalog.detach()),
            catalog_ce=float(ce.mean().detach()),top20_weight_mean=float(weights.mean()),
            top20_weight_min=float(weights.min()),top20_weight_max=float(weights.max()),
            top20_threshold_mean=float(beta.mean()),six_vector_regularizer=float(reg.detach()))
        return bpr+.05*catalog,reg

    def begin_evaluation(self):
        self._eval_embeddings=self.computer()

    def end_evaluation(self):
        if hasattr(self,'_eval_embeddings'):del self._eval_embeddings

    def getUsersRating(self, users):
        u,i=self._eval_embeddings if hasattr(self,'_eval_embeddings') else self.computer()
        return self.f(u[users.long()]@i.t())

    def evaluation_diagnostics(self):
        return {**self.latest_components,'catalog_auxiliary_weight':.05,'training_only_topk_cutoff':20}
