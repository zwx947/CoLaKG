"""Check dynamic expressivity, original-model bridge, Adam/RNG and ranking diagnostics."""
import copy
import importlib
import importlib.util
from pathlib import Path
import random
import sys
import types
import unittest
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'rec_code'))
world = types.ModuleType('world')
world.device, world.seed, world.cprint = torch.device('cpu'), 2020, lambda *args: None
world.config = {}
sys.modules['world'] = world
if importlib.util.find_spec('cppimport') is not None:
    raise RuntimeError('Refuse implicit optional extension compilation')
importlib.import_module('utils')  # retain original project's circular-import order
from model import CoLaKG
from research_methods.dynamic_semantic_attention import ResearchModel, dynamic_pair_logits


class FakeData:
    n_users, m_items = 4, 6

    def getSparseGraph(self):
        r = torch.tensor([[1., 1., 0., 0., 0., 0.], [0., 1., 1., 0., 0., 0.],
                          [0., 0., 0., 1., 1., 0.], [1., 0., 0., 0., 1., 1.]])
        a = torch.zeros(10, 10)
        a[:4, 4:], a[4:, :4] = r, r.t()
        inv = a.sum(1).pow(-.5)
        return (inv[:, None] * a * inv[None, :]).to_sparse().coalesce()


def fixture(cls=ResearchModel):
    torch.manual_seed(2020)
    config = dict(latent_dim_rec=64, lightGCN_n_layers=3, keep_prob=1., A_split=False,
                  pretrain=0, use_drop_edge=0, dropout_i=.6, dropout_u=.2, dropout_n=.6)
    neighbors = torch.tensor([[1, 2], [0, 2], [1, 3], [2, 4], [3, 5], [0, 4]])
    item, user = torch.randn(6, 1024) * .1, torch.randn(4, 1024) * .1
    return cls(config, FakeData(), neighbors, item, user)


U, P, N = torch.tensor([0, 1, 2, 3]), torch.tensor([0, 2, 4, 5]), torch.tensor([4, 3, 0, 2])


class Checks(unittest.TestCase):
    def test_query_can_reverse_a_shared_neighbor_ranking(self):
        centers = torch.tensor([[1., -1.], [-1., 1.]])
        neighbors = torch.tensor([[[1., -1.], [-1., 1.]]]).expand(2, -1, -1)
        actual = dynamic_pair_logits(neighbors, centers, torch.eye(2), torch.ones(2, 1))
        self.assertEqual(actual.argmax(1).tolist(), [0, 1])
        old_a = torch.tensor([[.1], [.3], [.2], [.4]])
        old = F.leaky_relu(torch.cat([neighbors, centers[:, None].expand_as(neighbors)], -1) @ old_a, .2).squeeze(-1)
        self.assertTrue(torch.equal(old[0].argsort(), old[1].argsort()))

    def test_static_bridge_matches_full_original_loss_and_gradients(self):
        original, candidate = fixture(CoLaKG), fixture()
        for name, value in original.state_dict().items():
            if name != 'a': self.assertTrue(torch.equal(value, candidate.state_dict()[name]), name)
        old_a = original.a.detach().clone()
        def static(self, neighbors, centers):
            key = neighbors @ self.W
            query = (centers @ self.W)[:, None].expand_as(key)
            return F.leaky_relu(torch.cat([key, query], -1) @ old_a, .2).squeeze(-1)
        candidate.pair_logits = types.MethodType(static, candidate)
        for training in [False, True]:
            original.train(training); candidate.train(training)
            torch.manual_seed(771); expected = original.computer()
            torch.manual_seed(771); actual = candidate.computer()
            for x, y in zip(expected, actual): torch.testing.assert_close(x, y)
            original.zero_grad(); candidate.zero_grad()
            torch.manual_seed(990); loss1, reg1 = original.bpr_loss(U, P, N)
            torch.manual_seed(990); loss2, reg2 = candidate.bpr_loss(U, P, N)
            torch.testing.assert_close(loss1, loss2); torch.testing.assert_close(reg1, reg2)
            (loss1 + 1e-4 * reg1).backward(); (loss2 + 1e-4 * reg2).backward()
            candidate_params = dict(candidate.named_parameters())
            for name, param in original.named_parameters():
                if name == 'a': continue
                other = candidate_params[name]
                if param.grad is None: self.assertIsNone(other.grad)
                else: torch.testing.assert_close(param.grad, other.grad)

    def test_active_parameters_update_and_exact_adam_rng_recovery(self):
        m = fixture().train(); opt = torch.optim.Adam(m.parameters(), lr=.001)
        def step(model, optimizer):
            optimizer.zero_grad(); loss, reg = model.bpr_loss(U, P, N)
            total = loss + 1e-4 * reg; total.backward()
            for name, param in model.named_parameters():
                if name in ['W_u', 'a_u']:
                    self.assertIsNone(param.grad)
                else:
                    self.assertIsNotNone(param.grad, name)
                    self.assertTrue(bool(torch.isfinite(param.grad).all()), name)
            self.assertGreater(float(model.W.grad.norm()), 0.)
            self.assertGreater(float(model.a.grad.norm()), 0.)
            optimizer.step(); return total.detach()
        before = m.a.detach().clone(); step(m, opt)
        self.assertFalse(torch.equal(before, m.a))
        state = copy.deepcopy({'model': m.state_dict(), 'optimizer': opt.state_dict(),
                    'torch': torch.get_rng_state(), 'numpy': np.random.get_state(), 'python': random.getstate()})
        expected = step(m, opt)
        recovered = fixture().train(); recovered.load_state_dict(state['model'])
        opt2 = torch.optim.Adam(recovered.parameters(), lr=.001); opt2.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch']); np.random.set_state(state['numpy']); random.setstate(state['python'])
        actual = step(recovered, opt2)
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(m.parameters(), recovered.parameters())))

    def test_ranking_cache_and_diagnostics_do_not_consume_rng(self):
        m = fixture().eval()
        rng = torch.get_rng_state().clone()
        with torch.no_grad():
            u, i = m.computer()
            m.begin_evaluation()
            actual = m.getUsersRating(torch.arange(4))
            torch.testing.assert_close(actual, torch.sigmoid(u @ i.t()))
            info = m.evaluation_diagnostics()
            for key in ['changed_neighbor_order_fraction_on_cyclic_query_replacement',
                        'changed_top1_fraction_on_cyclic_query_replacement', 'normalized_attention_entropy',
                        'mean_max_attention_weight', 'full_catalog_sigmoid_exact_one_fraction_including_known_positives']:
                self.assertTrue(0 <= info[key] <= 1, key)
            m.end_evaluation()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(hasattr(m, '_eval_embeddings'))


if __name__ == '__main__': unittest.main()
