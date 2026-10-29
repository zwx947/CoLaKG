"""Content-conditioned feature-wise center/neighbor fusion on fixed asymmetric attention."""
import torch
from torch import nn
from torch.nn import functional as F
from research_methods.asymmetric_semantic_attention import ResearchModel as Parent
from research_methods.asymmetric_semantic_attention import SOURCE_FILES as PARENT_SOURCES
SOURCE_FILES = [*PARENT_SOURCES, 'asymmetric_semantic_attention.py', '../content_neighbor_fusion_protocol.md']

class ResearchModel(Parent):
    def __init__(self, config, dataset, *args):
        super().__init__(config, dataset, *args)
        d = config['latent_dim_rec']
        self.content_gate = nn.Sequential(nn.Linear(4*d, 2*d), nn.ELU(), nn.Linear(2*d, d))
        self.method_config.update(name='content_neighbor_fusion', A_enabled=False,
            fusion='x + sigmoid(MLP([LN(x),LN(m),LN(m)-LN(x),LN(x)*LN(m)])) * (m-x)',
            gate_hidden=2*d, gate_output=d, gate_init='PyTorch Linear defaults; no identity constraint',
            gate_regularization='0.5 * sum squared gate parameters / minibatch size; original decay',
            adaptation='DCGL conditional integration and COHESION staged refinement inspiration; known gating, not new attention',
            degree_feature=False, extra_loss=False)

    def fuse_messages(self, center, message):
        x = F.layer_norm(center, (center.shape[-1],))
        m = F.layer_norm(message, (message.shape[-1],))
        gate = torch.sigmoid(self.content_gate(torch.cat([x,m,m-x,x*m], -1)))
        self._fusion_gate = gate.detach()
        return center + gate * (message-center)

    def bpr_loss(self, users, pos, neg):
        loss, reg = super().bpr_loss(users,pos,neg)
        gate_reg = .5 * sum(p.square().sum() for p in self.content_gate.parameters()) / len(users)
        self.latest_components['gate_regularizer'] = float(gate_reg.detach())
        return loss, reg + gate_reg

    def evaluation_diagnostics(self):
        out = super().evaluation_diagnostics()
        out.update(fusion_gate_mean=float(self._fusion_gate.mean()),
                   fusion_gate_std=float(self._fusion_gate.std(unbiased=False)),
                   fusion_gate_min=float(self._fusion_gate.min()),
                   fusion_gate_max=float(self._fusion_gate.max()))
        return out

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


        items_emb_merged = self.fuse_messages(items_emb_merged, h_prime)

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
