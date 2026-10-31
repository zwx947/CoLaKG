"""Check the joint operator, dense forward semantics and exact state recovery."""
import copy
import unittest
import numpy as np
import scipy.sparse as sp
import torch
from torch.nn import functional as F
from check_dynamic_semantic_attention import fixture,FakeData,U,P,N
from model import CoLaKG
from research_methods.joint_semantic_ssc import ResearchModel,build_operator

torch.set_num_threads(2)
FakeData.UserItemNet=sp.csr_matrix([[1.,1.,0.,0.,0.,0.],[0.,1.,1.,0.,0.,0.],
                                  [0.,0.,0.,1.,1.,0.],[1.,0.,0.,0.,1.,1.]])
class Checks(unittest.TestCase):
    def test_operator_symmetry_spectrum_and_rng(self):
        r=sp.csr_matrix([[1.,1.,0.,0.],[0.,0.,1.,1.]])
        neighbors=np.array([[1],[2],[3],[0]])
        state=np.random.get_state()
        t,meta=build_operator(r,neighbors)
        after=np.random.get_state()
        self.assertTrue(np.array_equal(state[1],after[1]));self.assertEqual(state[2],after[2])
        np.testing.assert_allclose(t.toarray(),t.T.toarray(),atol=1e-12)
        eigen=np.linalg.eigvalsh(t.toarray())
        self.assertAlmostEqual(eigen[0],-1.,places=7);self.assertAlmostEqual(eigen[-1],1.,places=7)
        self.assertEqual(meta['semantic_directed_entries'],8)

    def test_full_output_equals_independent_dense_formula_and_regularizer(self):
        base,m=fixture(CoLaKG),fixture(ResearchModel)
        m.eval();base.eval()
        u0=.5*(m.embedding_user.weight+F.elu(m.user_semantic_map(m.user_semantic_emb)))
        i0=.5*(m.embedding_item.weight+F.elu(m.semantic_map(m.semantic_emb)))
        first=torch.cat([u0,i0]);t=m.joint_operator.to_dense()
        expected=sum(torch.linalg.matrix_power(t,k)@first for k in range(4))/4
        torch.testing.assert_close(torch.cat(m.computer()),expected)
        _,ra=base.bpr_loss(U,P,N);_,rb=m.bpr_loss(U,P,N);torch.testing.assert_close(ra,rb)
        self.assertFalse(hasattr(m,'W'))

    def test_bpr_updates_maps_ids_and_restores_optimizer_rng(self):
        m=fixture(ResearchModel).train();o=torch.optim.Adam(m.parameters(),lr=.001)
        def step(model,opt):
            opt.zero_grad();loss,reg=model.bpr_loss(U,P,N);(loss+1e-4*reg).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            opt.step();return loss.detach()
        old=m.embedding_item.weight.detach().clone();step(m,o)
        self.assertFalse(torch.equal(old,m.embedding_item.weight))
        self.assertGreater(float(m.semantic_map.weight.grad.norm()),0)
        self.assertGreater(float(m.user_semantic_map.weight.grad.norm()),0)
        state=copy.deepcopy((m.state_dict(),o.state_dict(),torch.get_rng_state()));expected=step(m,o)
        r=fixture(ResearchModel).train();ro=torch.optim.Adam(r.parameters(),lr=.001)
        r.load_state_dict(state[0]);ro.load_state_dict(state[1]);torch.set_rng_state(state[2])
        self.assertTrue(torch.equal(expected,step(r,ro)))
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(m.parameters(),r.parameters())))
        r.eval();rng=torch.get_rng_state().clone()
        with torch.no_grad():
            u,i=r.computer();r.begin_evaluation()
            torch.testing.assert_close(r.getUsersRating(U),torch.sigmoid(u@i.T));r.end_evaluation()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

if __name__=='__main__':unittest.main()
