"""Component ablation of pair geometry: paired alignment without dispersion."""
from torch.nn import functional as F
from model import CoLaKG


def prepare_neighbours(dataset, item_semantics, original_neighbours):
    return original_neighbours


class ResearchModel(CoLaKG):
    def __init__(self, config, dataset, *args):
        super().__init__(config, dataset, *args)
        self.method_config={"ablation_of":"pair_geometry","removed_component":"same-type logsumexp dispersion",
                            "temperature":0.2,"auxiliary_weight":0.05,
                            "encoder_and_prediction":"original CoLaKG unchanged"}

    def bpr_loss(self, users, pos, neg):
        u,p,n,*regularized=self.getEmbedding(users.long(),pos.long(),neg.long())
        ranking=F.softplus((u*n).sum(1)-(u*p).sum(1)).mean()
        reg=sum(value.norm(2).pow(2) for value in regularized)/(2*len(users))
        alignment=-(F.normalize(u,dim=1)*F.normalize(p,dim=1)).sum(1).mean()/0.2
        return ranking+0.05*alignment,reg
