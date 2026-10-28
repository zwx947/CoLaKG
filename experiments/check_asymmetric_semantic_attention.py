"""Role-splitting bridge, asymmetric scores, new gradients and exact Adam recovery."""
import copy
import math
import random
import unittest
import numpy as np
import torch
from check_dynamic_semantic_attention import fixture, U, P, N
from research_methods.asymmetric_semantic_attention import ResearchModel, asymmetric_pair_logits


class Checks(unittest.TestCase):
    def test_constrained_roles_recover_shared_full_model_and_split_gradient(self):
        shared, new = fixture(), fixture(ResearchModel)
        with torch.no_grad():
            new.query_projection.copy_(new.W)
            new.attention_bias.zero_()
        for mode in [False, True]:
            shared.train(mode); new.train(mode)
            shared.zero_grad(); new.zero_grad()
            torch.manual_seed(134); expected = shared.computer()
            torch.manual_seed(134); actual = new.computer()
            for x, y in zip(expected, actual): torch.testing.assert_close(x, y)
            torch.manual_seed(181); l1, r1 = shared.bpr_loss(U, P, N)
            torch.manual_seed(181); l2, r2 = new.bpr_loss(U, P, N)
            torch.testing.assert_close(l1, l2); torch.testing.assert_close(r1, r2)
            (l1 + 1e-4 * r1).backward(); (l2 + 1e-4 * r2).backward()
            params = dict(new.named_parameters())
            for name, p in shared.named_parameters():
                if name == 'W':
                    torch.testing.assert_close(p.grad, new.W.grad + new.query_projection.grad)
                elif p.grad is None:
                    self.assertIsNone(params[name].grad)
                else:
                    torch.testing.assert_close(p.grad, params[name].grad)

    def test_center_neighbor_exchange_can_change_raw_score(self):
        i, j = torch.tensor([[1., -1.]]), torch.tensor([[-1., 1.]])
        key, query = torch.eye(2), torch.diag(torch.tensor([1., 2.]))
        bias, vector = torch.zeros(2), torch.ones(2, 1)
        forward = asymmetric_pair_logits(j[:, None], i, key, query, bias, vector)
        reverse = asymmetric_pair_logits(i[:, None], j, key, query, bias, vector)
        torch.testing.assert_close(forward, torch.tensor([[-.2]]))
        torch.testing.assert_close(reverse, torch.tensor([[1.]]))

    def test_new_parameters_update_and_adam_rng_continuation(self):
        model = fixture(ResearchModel).train()
        opt = torch.optim.Adam(model.parameters(), lr=.001)
        def step(m, o):
            o.zero_grad(); l, r = m.bpr_loss(U, P, N); total = l + 1e-4 * r
            total.backward()
            for name, p in m.named_parameters():
                if name in ['W_u', 'a_u']:
                    self.assertIsNone(p.grad)
                else:
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(bool(torch.isfinite(p.grad).all()), name)
            self.assertGreater(float(m.query_projection.grad.norm()), 0.)
            o.step(); return total.detach()
        before = model.query_projection.detach().clone()
        step(model, opt)
        self.assertFalse(torch.equal(before, model.query_projection))
        state = copy.deepcopy({'model': model.state_dict(), 'optimizer': opt.state_dict(),
            'torch': torch.get_rng_state(), 'numpy': np.random.get_state(), 'python': random.getstate()})
        expected = step(model, opt)
        recovered = fixture(ResearchModel).train(); recovered.load_state_dict(state['model'])
        opt2 = torch.optim.Adam(recovered.parameters(), lr=.001); opt2.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch']); np.random.set_state(state['numpy']); random.setstate(state['python'])
        actual = step(recovered, opt2)
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(model.parameters(), recovered.parameters())))

    def test_inherited_ranking_and_diagnostics_preserve_rng(self):
        m = fixture(ResearchModel).eval()
        rng = torch.get_rng_state().clone()
        with torch.no_grad():
            u, i = m.computer()
            m.begin_evaluation()
            torch.testing.assert_close(m.getUsersRating(torch.arange(4)), torch.sigmoid(u @ i.t()))
            info = m.evaluation_diagnostics()
            self.assertTrue(all(math.isfinite(x) for x in info.values() if isinstance(x, (int, float))))
            self.assertGreater(info['mean_absolute_forward_reverse_edge_logit_difference'], 0.)
            m.end_evaluation()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))


if __name__ == '__main__': unittest.main()
