"""Independent loss formula, parent bridge, shared gradient and full state recovery."""
import copy
import unittest
import numpy as np
import torch
from torch.nn import functional as F
from check_joint_semantic_ssc import fixture,U,P,N
from research_methods.joint_semantic_ssc import ResearchModel as Parent
from research_methods.semantic_history_completion import ResearchModel,completion_loss


class Checks(unittest.TestCase):
    def test_formula_permutation_mask_and_shared_gradient(self):
        m=fixture(ResearchModel).double()
        sem=m.semantic_emb.double();targets=m.completion_targets.double()
        visible=[[0,1],[3,4]];hidden=[[2,5],[0,2]]
        loss,q=completion_loss(m.semantic_map,sem,targets,visible,hidden)
        values=[]
        for v,h in zip(visible,hidden):
            a=F.elu(F.linear(sem[v],m.semantic_map.weight,m.semantic_map.bias)).mean(0)
            pred=F.normalize(m.semantic_map.weight.T@a,dim=0)
            logits=pred@targets.T/.2
            candidates=[i for i in range(6) if i not in v]
            values.append(torch.logsumexp(logits[candidates],0)-logits[h].mean())
        torch.testing.assert_close(loss,torch.stack(values).mean())
        rev=completion_loss(m.semantic_map,sem,targets,[row[::-1] for row in visible],hidden)[0]
        torch.testing.assert_close(loss,rev)
        loss.backward()
        self.assertGreater(float(m.semantic_map.weight.grad.norm()),0)
        self.assertGreater(float(m.semantic_map.bias.grad.norm()),0)
        self.assertIsNone(m.embedding_item.weight.grad)
        self.assertFalse(targets.requires_grad)
        old=m.semantic_map.weight[0,0].item();eps=1e-6
        with torch.no_grad():m.semantic_map.weight[0,0]=old+eps
        plus=completion_loss(m.semantic_map,sem,targets,visible,hidden)[0].item()
        with torch.no_grad():m.semantic_map.weight[0,0]=old-eps
        minus=completion_loss(m.semantic_map,sem,targets,visible,hidden)[0].item()
        self.assertAlmostEqual((plus-minus)/(2*eps),m.semantic_map.weight.grad[0,0].item(),places=5)

    def test_parent_bridge_and_rng_isolation(self):
        p,m=fixture(Parent),fixture(ResearchModel)
        m.aux_weight=0
        for training in [False,True]:
            p.train(training);m.train(training)
            torch.manual_seed(90);a,ar=p.bpr_loss(U,P,N)
            torch.manual_seed(90);b,br=m.bpr_loss(U,P,N)
            torch.testing.assert_close(a,b);torch.testing.assert_close(ar,br)
        cpu=torch.get_rng_state().clone();state=np.random.get_state()
        v,h=m.sample_completion()
        self.assertTrue(torch.equal(cpu,torch.get_rng_state()))
        self.assertTrue(np.array_equal(state[1],np.random.get_state()[1]))
        for visible,hidden in zip(v,h):
            self.assertFalse(set(visible)&set(hidden));self.assertTrue(visible and hidden)
        self.assertEqual(sum(x.numel() for x in p.parameters()),sum(x.numel() for x in m.parameters()))

    def test_optimizer_and_aux_rng_restored(self):
        m=fixture(ResearchModel).train();opt=torch.optim.Adam(m.parameters(),lr=.001)
        def step(net,o):
            o.zero_grad();loss,reg=net.bpr_loss(U,P,N);(loss+1e-4*reg).backward()
            self.assertTrue(all(torch.isfinite(x.grad).all() for x in net.parameters() if x.grad is not None))
            o.step();return loss.detach()
        step(m,opt)
        state=copy.deepcopy((m.state_dict(),opt.state_dict(),torch.get_rng_state()))
        expected=step(m,opt)
        n=fixture(ResearchModel).train();no=torch.optim.Adam(n.parameters(),lr=.001)
        n.load_state_dict(state[0]);no.load_state_dict(state[1]);torch.set_rng_state(state[2])
        self.assertTrue(torch.equal(expected,step(n,no)))
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(m.parameters(),n.parameters())))
        self.assertEqual(m.sample_completion(),n.sample_completion())

if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
