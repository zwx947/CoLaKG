"""GATv2-style pair scoring inside CoLaKG's existing semantic-neighbor module.

Uses the author implementation's share_weights=True, single-head scoring form,
without PyG dependencies. It does not replace CoLaKG's values or UI propagation.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from model import CoLaKG

SOURCE_FILES = ['../dynamic_semantic_attention_protocol.md',
                '../research_records/dynamic_semantic_attention_sources_20261028.json']


def dynamic_pair_logits(neighbors, centers, projection, vector):
    key = neighbors @ projection
    query = (centers @ projection).unsqueeze(1)
    return (F.leaky_relu(key + query, negative_slope=.2) @ vector).squeeze(-1)


class ResearchModel(CoLaKG):
    requires_custom_ranking = False

    def __init__(self, config, dataset, *args):
        super().__init__(config, dataset, *args)
        # W is the unchanged shared 1024 -> 32 projection. Only the scoring
        # vector changes shape because the nonlinearity precedes the final sum.
        self.a = nn.Parameter(torch.empty(self.semantic_hid, 1))
        nn.init.xavier_uniform_(self.a, gain=1.414)
        self.latest_components = {}
        self.method_config = {
            'name': 'dynamic_semantic_attention', 'A_enabled': False,
            'original_logits': 'LeakyReLU([s_neighbor W || s_center W] a64)',
            'new_logits': 'a32^T LeakyReLU(s_neighbor W + s_center W)',
            'shared_query_key_projection': True, 'attention_hidden_dim': self.semantic_hid,
            'heads': 1, 'bias': False, 'negative_slope': .2,
            'scorer_initializer': 'original Xavier gain 1.414, fresh recommendation parameters',
            'neighbors': 'unchanged sklearn cosine argsort[:,1:11], fixed original top10, no added self loops',
            'attention_dropout': self.dropout_neighbor,
            'values': 'unchanged original fused item ID/semantic vectors; no new value projection',
            'input_fusion': 'original user AND item Linear/ELU/dropout, 1:1 mean with ID',
            'propagation': 'unchanged C value aggregation and 3-layer original UI LightGCN with layer mean',
            'objective': 'unchanged BPR and original six-vector L2; same optimizer and negative sampler',
            'legacy_unused_parameters': ['W_u', 'a_u'],
            'parameter_delta_vs_original': -32,
            'source': 'How Attentive are Graph Attention Networks?, ICLR2022; author GATv2Conv share_weights option',
            'source_commit': '6f6d29089603c2f51651bd083dcd44f206b1100e',
            'adaptation': 'semantic pair scorer only; original recommendation values/graph/inputs retained; no PyG/DGL installation',
            'scope': 'known dynamic attention applied to this semantic retrieval module; not a claim of new attention',
        }

    def pair_logits(self, neighbors, centers):
        return dynamic_pair_logits(neighbors, centers, self.W, self.a)

    def begin_evaluation(self):
        self._score_count = 0
        self._saturated_one = 0
        self._logit_max = -math.inf
        self._logit_min = math.inf
        self._eval_embeddings = self.computer()

    def end_evaluation(self):
        for name in ['_eval_embeddings', '_eval_attention', '_eval_attention_logits']:
            if hasattr(self, name):
                delattr(self, name)

    def getUsersRating(self, users):
        u, i = self._eval_embeddings if hasattr(self, '_eval_embeddings') else self.computer()
        logits = u[users.long()] @ i.t()
        scores = self.f(logits)
        if hasattr(self, '_eval_embeddings'):
            self._score_count += scores.numel()
            self._saturated_one += int((scores == 1).sum())
            self._logit_max = max(self._logit_max, float(logits.max()))
            self._logit_min = min(self._logit_min, float(logits.min()))
        return scores

    def evaluation_diagnostics(self):
        weights = self._eval_attention
        # Fixed cyclic query replacement consumes no RNG; neighbors stay fixed.
        alternate = self.pair_logits(self.semantic_emb[self.adj_matrix], self.semantic_emb.roll(1, 0))
        original_order = self._eval_attention_logits.argsort(1, descending=True)
        alternate_order = alternate.argsort(1, descending=True)
        return {
            'changed_neighbor_order_fraction_on_cyclic_query_replacement': float((original_order != alternate_order).any(1).float().mean()),
            'changed_top1_fraction_on_cyclic_query_replacement': float((original_order[:, 0] != alternate_order[:, 0]).float().mean()),
            'normalized_attention_entropy': float(-(weights * weights.clamp_min(1e-30).log()).sum(1).mean() / math.log(weights.shape[1])),
            'mean_max_attention_weight': float(weights.max(1).values.mean()),
            'full_catalog_sigmoid_exact_one_fraction_including_known_positives': self._saturated_one / self._score_count,
            'full_catalog_logit_min': self._logit_min, 'full_catalog_logit_max': self._logit_max,
            'interpretation': 'query dependence and numerical diagnostics only; no method selection or test-label use inside the encoder',
        }

    def bpr_loss(self, users, pos, neg):
        loss, regularizer = super().bpr_loss(users, pos, neg)
        self.latest_components = {'bpr': float(loss.detach()), 'six_vector_regularizer': float(regularizer.detach())}
        return loss, regularizer

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        items_semantic_emb = F.dropout(self.semantic_emb, self.dropout_i, training=self.training)
        items_semantic_emb = self.semantic_map(items_semantic_emb)
        items_semantic_emb = F.elu(items_semantic_emb)
        items_semantic_emb = F.dropout(items_semantic_emb, self.dropout_i, training=self.training)
        items_emb_merged = (items_emb + items_semantic_emb) / 2

        user_semantic_emb = F.dropout(self.user_semantic_emb, self.dropout_u, training=self.training)
        user_semantic_emb = self.user_semantic_map(user_semantic_emb)
        user_semantic_emb = F.elu(user_semantic_emb)
        user_semantic_emb = F.dropout(user_semantic_emb, self.dropout_u, training=self.training)
        users_emb_merged = (users_emb + user_semantic_emb) / 2


        neighbor_emb = items_emb_merged[self.adj_matrix]
        items_semantic_emb0 = self.semantic_emb
        neighbor_semantic_emb = self.semantic_emb[self.adj_matrix]  # N,L,d1

        # x = self.attentions(neighbor_semantic_emb, neighbor_emb, items_semantic_emb0)
        h, value_emb, semantic_emb = neighbor_semantic_emb, neighbor_emb, items_semantic_emb0

        attention = self.pair_logits(h, semantic_emb)
        if not self.training:
            self._eval_attention_logits = attention.detach()
        attention = F.softmax(attention, dim=1) # N,L
        if not self.training:
            self._eval_attention = attention.detach()

        attention = F.dropout(attention, self.dropout_neighbor, training=self.training) # N,L
        attention = attention.unsqueeze(-1)

        h_prime = attention * value_emb

        h_prime = torch.sum(h_prime, dim=1)

        h_prime = F.elu(h_prime)


        items_emb_merged = (items_emb_merged + h_prime ) / 2

        # items_emb = F.elu(items_emb)

        all_emb = torch.cat([users_emb_merged, items_emb_merged])
        embs = [all_emb]

        if self.config['use_drop_edge']:
            if self.training:
                # print("droping")
                g_droped = self._CoLaKG__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
