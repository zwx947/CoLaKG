"""Numerics, full-forward bridge, updates and exact Adam/RNG restoration."""
import copy
import types
import unittest
import numpy as np
import torch
from check_dynamic_semantic_attention import fixture, FakeData, U, P, N
from research_methods.asymmetric_semantic_attention import ResearchModel as Parent
from research_methods.calibrated_neighbor_fusion import ResearchModel

FakeData.UserItemNet = np.ones((4,6), dtype=np.float32)


class Checks(unittest.TestCase):
    def test_unchanged_forward_when_fusion_bypassed(self):
        a,b = fixture(Parent),fixture(ResearchModel)
        b.fuse_messages = types.MethodType(lambda self,x,m: .5*(x+m),b)
        for training in [False,True]:
            a.train(training);b.train(training)
            torch.manual_seed(18);left = a.computer()
            torch.manual_seed(18);right = b.computer()
            for x,y in zip(left,right):torch.testing.assert_close(x,y)

    def test_norm_control_gate_variation_and_zero_safety(self):
        m = fixture(ResearchModel).eval()
        x,y = torch.randn(6,64),torch.randn(6,64)*.2
        scale = ((x+y)*.5).norm(dim=-1)
        with torch.no_grad():
            z=m.fuse_messages(x,y)
            torch.testing.assert_close(z.norm(dim=-1),scale)
            self.assertEqual(m._fusion_gate.shape,(6,1))
            before=z.clone();m.item_gate[-1].bias.add_(3)
            after=m.fuse_messages(x,y)
            self.assertFalse(torch.allclose(before,after))
            torch.testing.assert_close(after.norm(dim=-1),scale)
            zeros=m.fuse_messages(torch.zeros_like(x),torch.zeros_like(y))
            self.assertTrue(torch.equal(zeros,torch.zeros_like(zeros)))

    def test_ranking_gradients_updates_and_rng_restore(self):
        m=fixture(ResearchModel).train();opt=torch.optim.Adam(m.parameters(),lr=.001)
        def step(model,optim):
            optim.zero_grad();loss,reg=model.bpr_loss(U,P,N)
            (loss+1e-4*reg).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            optim.step();return loss.detach()
        loss,_=m.bpr_loss(U,P,N);loss.backward()
        for module in [m.item_gate,m.message_adapter]:
            self.assertGreater(float(module[0].weight.grad.norm()),0)
        previous=[m.item_gate[0].weight.detach().clone(),m.message_adapter[0].weight.detach().clone()]
        step(m,opt)
        for old,now in zip(previous,[m.item_gate[0].weight,m.message_adapter[0].weight]):
            self.assertFalse(torch.equal(old,now))
        saved=copy.deepcopy((m.state_dict(),opt.state_dict(),torch.get_rng_state()))
        expected=step(m,opt)
        r=fixture(ResearchModel);o=torch.optim.Adam(r.parameters(),lr=.001)
        r.load_state_dict(saved[0]);o.load_state_dict(saved[1]);torch.set_rng_state(saved[2])
        actual=step(r,o)
        self.assertTrue(torch.equal(expected,actual))
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(m.parameters(),r.parameters())))
        r.eval();rng=torch.get_rng_state().clone()
        with torch.no_grad():
            u,i=r.computer();r.begin_evaluation()
            torch.testing.assert_close(r.getUsersRating(torch.arange(4)),torch.sigmoid(u@i.T))
            self.assertLess(r.evaluation_diagnostics()['fusion_norm_max_error'],1e-5)
            r.end_evaluation()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))


if __name__=='__main__':unittest.main()
