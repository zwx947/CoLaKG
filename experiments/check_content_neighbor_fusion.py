import copy
import types
import unittest
import torch
from check_dynamic_semantic_attention import fixture, U, P, N
from research_methods.asymmetric_semantic_attention import ResearchModel as Parent
from research_methods.content_neighbor_fusion import ResearchModel

class Checks(unittest.TestCase):
    def test_half_gate_recovers_parent_outputs_and_gradients(self):
        a, b = fixture(Parent), fixture(ResearchModel)
        b.fuse_messages = types.MethodType(lambda self,x,m: x + .5*(m-x), b)
        for training in [False, True]:
            a.train(training); b.train(training)
            torch.manual_seed(18); x=a.computer()
            torch.manual_seed(18); y=b.computer()
            for left,right in zip(x,y): torch.testing.assert_close(left,right)
            a.zero_grad(); b.zero_grad()
            torch.manual_seed(23); la,ra=a.bpr_loss(U,P,N)
            torch.manual_seed(23); lb,rb=b.bpr_loss(U,P,N)
            torch.testing.assert_close(la,lb)
            la.backward(); lb.backward()
            bp=dict(b.named_parameters())
            for name,p in a.named_parameters():
                if p.grad is not None: torch.testing.assert_close(p.grad,bp[name].grad)

    def test_gate_learns_from_ranking_and_adam_rng_restores(self):
        m=fixture(ResearchModel).train(); opt=torch.optim.Adam(m.parameters(),lr=.001)
        def step(model,optim):
            optim.zero_grad(); loss,reg=model.bpr_loss(U,P,N)
            (loss+1e-4*reg).backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            optim.step(); return loss.detach()
        loss,_=m.bpr_loss(U,P,N);loss.backward()
        self.assertGreater(float(m.content_gate[0].weight.grad.norm()),0)
        old=m.content_gate[0].weight.detach().clone();step(m,opt)
        self.assertFalse(torch.equal(old,m.content_gate[0].weight))
        saved=copy.deepcopy((m.state_dict(),opt.state_dict(),torch.get_rng_state()))
        expected=step(m,opt)
        r=fixture(ResearchModel);o=torch.optim.Adam(r.parameters(),lr=.001)
        r.load_state_dict(saved[0]);o.load_state_dict(saved[1]);torch.set_rng_state(saved[2])
        actual=step(r,o)
        self.assertTrue(torch.equal(expected,actual))
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(m.parameters(),r.parameters())))

    def test_ranking_cache_and_gate_diagnostics(self):
        m=fixture(ResearchModel).eval();rng=torch.get_rng_state().clone()
        with torch.no_grad():
            u,i=m.computer();m.begin_evaluation()
            torch.testing.assert_close(m.getUsersRating(torch.arange(4)),torch.sigmoid(u@i.T))
            d=m.evaluation_diagnostics();self.assertGreater(d['fusion_gate_std'],0)
            self.assertGreater(d['fusion_gate_min'],0);self.assertLess(d['fusion_gate_max'],1)
            m.end_evaluation()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

if __name__=='__main__': unittest.main()
