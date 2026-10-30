"""Center-conditioned message adaptation and norm-controlled item-wise fusion."""
import torch
from torch import nn
from torch.nn import functional as F
from research_methods.asymmetric_semantic_attention import ResearchModel as Parent
from research_methods.content_neighbor_fusion import ResearchModel as Forward
from research_methods.content_neighbor_fusion import SOURCE_FILES as FORWARD_SOURCES

SOURCE_FILES = [*FORWARD_SOURCES, 'content_neighbor_fusion.py',
                '../calibrated_neighbor_fusion_protocol.md']


class ResearchModel(Parent):
    # Reuse the already checked CoLaKG forward; its fusion call is overridden below.
    computer = Forward.computer

    def __init__(self, config, dataset, *args):
        super().__init__(config, dataset, *args)
        d = config['latent_dim_rec']
        self.message_adapter = nn.Sequential(nn.Linear(4*d, d), nn.ELU(), nn.Linear(d, d))
        self.item_gate = nn.Sequential(nn.Linear(4*d+1, d), nn.ELU(), nn.Linear(d, 1))
        # Strictly train-only support count, without a prescribed monotone effect.
        counts = torch.as_tensor(dataset.UserItemNet.sum(axis=0)).float().flatten()
        self.register_buffer('log_train_support', counts.log1p()[:, None])
        self.method_config.update(name='calibrated_neighbor_fusion', A_enabled=False,
            message_adapter='MLP([unit(x),unit(m),difference,product]) 256->64->64 ELU',
            gate='sigmoid MLP([unit(x),unit(adapted_m),difference,product,log1p(train_count)]) 257->64->1',
            fusion='norm((x+m)/2) * unit((1-g)*unit(x)+g*unit(adapted_m))',
            normalization_epsilon=1e-6, initialization='PyTorch Linear defaults, fresh random',
            extra_loss=False, gate_balance_penalty=False,
            regularization='adapter+gate 0.5 squared parameter norm / actual batch; original decay',
            sources='COHESION staged refinement and DCGL conditional fusion; norm control is a task adaptation hypothesis, not a proven novelty or calibrated reliability')

    @staticmethod
    def unit(value):
        return F.normalize(value, p=2, dim=-1, eps=1e-6)

    @staticmethod
    def features(x, m):
        return torch.cat([x, m, m-x, x*m], dim=-1)

    def fuse_messages(self, center, message):
        x, m = self.unit(center), self.unit(message)
        adapted = self.message_adapter(self.features(x, m))
        adapted_unit = self.unit(adapted)
        gate_input = torch.cat([self.features(x, adapted_unit), self.log_train_support], dim=-1)
        gate = torch.sigmoid(self.item_gate(gate_input))
        mixed = (1-gate)*x + gate*adapted_unit
        # Retain the parent's per-item magnitude, differentiable in the original values.
        scale = ((center+message)*.5).norm(dim=-1, keepdim=True)
        output = scale * self.unit(mixed)
        self._fusion_gate = gate.detach()
        if not self.training:
            self._norm_diagnostics = dict(
                fusion_output_norm_mean=float(output.norm(dim=-1).mean()),
                fusion_reference_norm_mean=float(scale.mean()),
                fusion_norm_max_error=float((output.norm(dim=-1)-scale.flatten()).abs().max()),
                fusion_mixed_near_zero_count=int((mixed.norm(dim=-1)<1e-6).sum()),
                adapter_near_zero_count=int((adapted.norm(dim=-1)<1e-6).sum()))
        return output

    def bpr_loss(self, users, pos, neg):
        loss, reg = super().bpr_loss(users, pos, neg)
        extra = .5*sum(p.square().sum() for module in [self.message_adapter,self.item_gate]
                       for p in module.parameters())/len(users)
        self.latest_components['fusion_regularizer'] = float(extra.detach())
        return loss, reg+extra

    def evaluation_diagnostics(self):
        out = super().evaluation_diagnostics()
        gate = self._fusion_gate
        out.update(self._norm_diagnostics)
        out.update(fusion_gate_mean=float(gate.mean()),
                   fusion_gate_std=float(gate.std(unbiased=False)),
                   fusion_gate_min=float(gate.min()), fusion_gate_max=float(gate.max()),
                   fusion_gate_below_001=float((gate<.01).float().mean()),
                   fusion_gate_above_099=float((gate>.99).float().mean()))
        return out
