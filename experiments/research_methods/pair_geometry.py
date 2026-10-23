"""Original BPR plus SGCL-inspired same-type pair geometry regularization.

Unlike catalog negatives, the denominator disperses user-user and item-item
representations. This is a transfer of an existing objective, not a novelty claim.
"""
import torch
from torch.nn import functional as F
from model import CoLaKG

TEMPERATURE=0.2
GEOMETRY_WEIGHT=0.05


def prepare_neighbours(dataset, item_semantics, original_neighbours):
    return original_neighbours


class ResearchModel(CoLaKG):
    def __init__(self, config, dataset, *args):
        super().__init__(config, dataset, *args)
        self.method_config={"geometry":"logsumexp(user-user AND positive-item-positive-item) minus paired cosine",
                            "temperature":TEMPERATURE,"auxiliary_weight":GEOMETRY_WEIGHT,
                            "encoder_and_prediction":"original CoLaKG unchanged",
                            "diagonal_and_duplicates":"included, as in public SGCL form",
                            "normalization":"mean auxiliary loss, not official standalone summed loss",
                            "attribution":"SGCL RecSys2025, DavidZWZ/SGCL; adapted as BPR auxiliary"}

    def bpr_loss(self, users, pos, neg):
        u,p,n,*regularized=self.getEmbedding(users.long(),pos.long(),neg.long())
        ranking=F.softplus((u*n).sum(1)-(u*p).sum(1)).mean()
        reg=sum(value.norm(2).pow(2) for value in regularized)/(2*len(users))
        normalized_u,normalized_p=F.normalize(u,dim=1),F.normalize(p,dim=1)
        within_types=torch.cat([normalized_u@normalized_u.t(),normalized_p@normalized_p.t()],dim=1)
        positive=(normalized_u*normalized_p).sum(1)
        geometry=(torch.logsumexp(within_types/TEMPERATURE,dim=1)-positive/TEMPERATURE).mean()
        return ranking+GEOMETRY_WEIGHT*geometry,reg
