"""Independent center/neighbor projections and a learned nonlinear threshold.

An evolution of the fixed shared-projection dynamic scorer; no A or extra loss.
"""
import torch
from torch import nn
from torch.nn import functional as F
from research_methods.dynamic_semantic_attention import ResearchModel as SharedModel
from research_methods.dynamic_semantic_attention import SOURCE_FILES as SHARED_SOURCES

SOURCE_FILES = [*SHARED_SOURCES, 'dynamic_semantic_attention.py',
                '../asymmetric_semantic_attention_protocol.md',
                '../research_records/asymmetric_semantic_attention_sources_20261028.json']


def asymmetric_pair_logits(neighbors, centers, key_projection, query_projection, bias, vector):
    key = neighbors @ key_projection
    query = (centers @ query_projection).unsqueeze(1)
    return (F.leaky_relu(key + query + bias, negative_slope=.2) @ vector).squeeze(-1)


class ResearchModel(SharedModel):
    def __init__(self, config, dataset, *args):
        super().__init__(config, dataset, *args)
        self.query_projection = nn.Parameter(torch.empty_like(self.W))
        nn.init.xavier_uniform_(self.query_projection, gain=1.414)
        self.attention_bias = nn.Parameter(torch.zeros(self.semantic_hid))
        self.method_config.update({
            'name': 'asymmetric_semantic_attention',
            'parent_method': 'dynamic_semantic_attention, fixed code a1e2693/result 5c5ee01',
            'new_logits': 'a32^T LeakyReLU(s_neighbor K + s_center Q + b32)',
            'shared_query_key_projection': False, 'bias': True,
            'query_initializer': 'independent Xavier gain 1.414',
            'bias_initializer': 'zeros; learned before the LeakyReLU threshold',
            'parameter_delta_vs_original': 32768,
            'parameter_delta_vs_shared_dynamic': 32800,
            'adaptation': 'author-supported independent query/key scoring and preactivation bias; CoLaKG values, candidates, inputs, UI graph, BPR and regularization retained',
            'motivation': 'center and supporting neighbor have different roles; shared sum forced symmetric unnormalized edge scores',
            'attention_sharpness_objective': False,
            'novelty_boundary': 'known GATv2 choices adapted to the semantic retrieval scorer, no new attention claim',
        })

    def pair_logits(self, neighbors, centers):
        return asymmetric_pair_logits(neighbors, centers, self.W, self.query_projection, self.attention_bias, self.a)

    def evaluation_diagnostics(self):
        result = super().evaluation_diagnostics()
        key = self.semantic_emb @ self.W
        query = self.semantic_emb @ self.query_projection
        reverse = (F.leaky_relu(query[self.adj_matrix] + key[:, None, :] + self.attention_bias, .2) @ self.a).squeeze(-1)
        result.update({
            'mean_absolute_forward_reverse_edge_logit_difference': float((self._eval_attention_logits - reverse).abs().mean()),
            'query_key_projection_cosine': float(F.cosine_similarity(self.W.flatten(), self.query_projection.flatten(), dim=0)),
            'attention_bias_l2': float(self.attention_bias.norm()),
        })
        return result
