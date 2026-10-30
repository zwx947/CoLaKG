"""Check the objective's semantics and complete original-encoder integration."""
import copy
import unittest
import numpy as np
import torch
from torch.nn import functional as F
from check_dynamic_semantic_attention import ROOT
from model import CoLaKG
from research_methods.top20_catalog_ranking import ResearchModel,catalog_terms

torch.set_num_threads(2)
def fixture(cls):
    torch.manual_seed(2020)
    net=np.zeros((4,32),dtype=np.float32)
    for u,items in enumerate([[0,1],[2,3],[4,6],[5,7]]):net[u,items]=1
    r=torch.tensor(net);a=torch.zeros(36,36);a[:4,4:]=r;a[4:,:4]=r.T
    deg=a.sum(1).clamp_min(1).rsqrt();graph=(deg[:,None]*a*deg[None,:]).to_sparse().coalesce()
    class Data:
        n_users,m_items=4,32
        UserItemNet=__import__('scipy').sparse.csr_matrix(net)
        def getSparseGraph(self):return graph
    config=dict(latent_dim_rec=64,lightGCN_n_layers=3,keep_prob=1.,A_split=False,pretrain=0,use_drop_edge=0,dropout_i=.6,dropout_u=.2,dropout_n=.6)
    semantic_i,semantic_u=torch.randn(32,1024)*.1,torch.randn(4,1024)*.1
    neighbors=torch.stack([(torch.arange(32)+1)%32,(torch.arange(32)+2)%32],1)
    return cls(config,Data(),neighbors,semantic_i,semantic_u)

U,P,N=torch.arange(4),torch.tensor([0,2,4,5]),torch.tensor([20,21,22,23])
class Checks(unittest.TestCase):
    def test_mask_and_threshold_and_exact_ce(self):
        scores=torch.tensor([[2.,1000.,3.,4.,5.]],requires_grad=True)
        mask=torch.tensor([[True,True,False,False,False]])
        loss,w,beta,ce=catalog_terms(scores,torch.tensor([0]),mask,k=2)
        torch.testing.assert_close(beta,torch.tensor([4.]))
        torch.testing.assert_close(ce,F.cross_entropy(scores[:,[0,2,3,4]],torch.tensor([0]),reduction='none'))
        self.assertFalse(w.requires_grad);self.assertFalse(beta.requires_grad)
        loss.backward()
        self.assertLess(float(scores.grad[0,0]),0)
        self.assertEqual(float(scores.grad[0,1]),0)
        self.assertTrue((scores.grad[0,2:]>0).all())

    def test_positive_dominance_remains_finite_nonnegative(self):
        scores=torch.tensor([[1000.,-1000.,-1000.]],requires_grad=True)
        loss,*_=catalog_terms(scores,torch.tensor([0]),torch.tensor([[True,False,False]]),k=1)
        loss.backward();self.assertGreaterEqual(float(loss),0)
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_original_encoder_regularizer_and_rng_bridge(self):
        base,new=fixture(CoLaKG),fixture(ResearchModel)
        self.assertEqual(sum(p.numel() for p in base.parameters()),sum(p.numel() for p in new.parameters()))
        for training in [False,True]:
            base.train(training);new.train(training)
            torch.manual_seed(99);a=base.computer()
            torch.manual_seed(99);b=new.computer()
            for x,y in zip(a,b):torch.testing.assert_close(x,y)
            torch.manual_seed(11);_,ar=base.bpr_loss(U,P,N)
            torch.manual_seed(11);_,br=new.bpr_loss(U,P,N)
            torch.testing.assert_close(ar,br)

    def test_optimizer_restore_and_ranking(self):
        m=fixture(ResearchModel).train();o=torch.optim.Adam(m.parameters(),lr=.001)
        def step(model,opt):
            opt.zero_grad();loss,reg=model.bpr_loss(U,P,N);(loss+1e-4*reg).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            opt.step();return loss.detach()
        before=m.embedding_item.weight.detach().clone();step(m,o)
        self.assertFalse(torch.equal(before,m.embedding_item.weight))
        saved=copy.deepcopy((m.state_dict(),o.state_dict(),torch.get_rng_state()));expected=step(m,o)
        r=fixture(ResearchModel).train();ro=torch.optim.Adam(r.parameters(),lr=.001)
        r.load_state_dict(saved[0]);ro.load_state_dict(saved[1]);torch.set_rng_state(saved[2])
        self.assertTrue(torch.equal(expected,step(r,ro)))
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(m.parameters(),r.parameters())))
        r.eval();rng=torch.get_rng_state().clone()
        with torch.no_grad():
            u,i=r.computer();r.begin_evaluation()
            torch.testing.assert_close(r.getUsersRating(U),torch.sigmoid(u@i.T));r.end_evaluation()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

if __name__=='__main__':unittest.main()
