import unittest
import torch
from models.SimBa_hybrid_control import HybridActionGMM, HybridGMMMixin, HybridControlSimBaActor
from torch.distributions import MixtureSameFamily, Normal, Bernoulli, Independent, Categorical
from torch.distributions.distribution import Distribution

class TestHybridActionGMM(unittest.TestCase):

    def setUp(self):
        self.batch_size = 3
        # define the component distribution
        means = torch.tensor(
            [[0,1],
             [0,1],
             [0,1],
             [0,1],
             [0,1],
             [0,1]], dtype=torch.float32
        )
        means = means.unsqueeze(0).repeat(self.batch_size,1,1)
        std = torch.tensor(
            [[1,0.5],
             [1,0.5],
             [1,0.5],
             [1,0.5],
             [1,0.5],
             [1,0.5]]
        )
        std = std.unsqueeze(0).repeat(self.batch_size,1,1)
        components = Normal(loc=means, scale=std)

        # define the mixture distribution
        logits = torch.ones((6,2))
        logits[:,0] = 0.75
        logits[:,1] = 1 - logits[:,0]
        logits = logits.unsqueeze(0).repeat(self.batch_size,1,1)

        mix_dist = Categorical(probs=logits)
        
        
        # define GMM for testing
        self.gmm = HybridActionGMM(
            mix_dist,
            components,
            force_weight=1.0,
            pos_weight = 1.0,
            rot_weight=1.0,
            torque_weight=1.0,
            ctrl_torque = True,
            uniform_rate = 0.01
        )

    def tearDown(self):
        pass

    def gumble_rate(self, uniform_rate = 0.01, n=100):
        self.gmm.uniform_rate = uniform_rate
        x1 = 0
        x0 = 0
        for i in range(n):
            y = torch.where(self.gmm.sample_gumbel_softmax(self.gmm.mixture_distribution.probs.log()) > 0.5, 1, 0)
            x0 += torch.sum(y[:,:,0])
            x1 += torch.sum(y[:,:,1])

        p0 = x0 / (n * 6 * self.batch_size)
        p1 = x1 / (n * 6 * self.batch_size)
        #print("\n\n",p0, p1, "\n\n")
        return p0,p1
    
    def test_sample_gumbel_softmax_no_uniform(self):
        #print(self.gmm.mixture_distribution.probs)
        p0, p1 = self.gumble_rate(0.00, 1000)
        self.assertTrue( p0 > 0.73 and p0 < 0.77)
        self.assertTrue( p1 > 0.23 and p1 < 0.27)

    def test_sample_gumbel_softmax_uniform(self):
        p0, p1 = self.gumble_rate(1.0, 1000)
        self.assertTrue( p0 > 0.48 and p0 < 0.52)
        self.assertTrue( p1 > 0.48 and p1 < 0.52)
    
    def test_log_prob(self):
        self.batch_size=3
        from math import log
        test_case = torch.tensor(
            [[ 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,

               0.0, 0.5, 0.5, 1.0, 1.0, 0.0,
               0.0, 0.5, 0.5, 1.0, 1.0, 0.0], # same value

             [ 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,

               0.0, 0.0, 0.5, 0.5, 0.0, 0.0,
               1.0, 1.0, 1.0, 1.0, 0.5, 0.5],

             [ 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,

               1.0, 1.0, 1.0, 1.0, 0.5, 0.5,
               0.5, 0.5, 0.0, 0.0, 0.0, 0.0]]
        )

        probs = {
            0:[0.398942,0.107982], #0.0
            0.5:[0.352065,0.483941], #0.5
            1:[0.241971,0.797885] #1.0
        }

        log_probs = torch.zeros((self.batch_size, 6))
        for i in range(3):
            s = torch.where(test_case[i,:6] > 0.5, 0.75, 0.25)
            sp = 1 - s
            
            for k in range(6):
                p = probs[test_case[i,6+k].item()][0]
                f = probs[test_case[i,12+k].item()][1]
                log_probs[i,k] = log( 0.75 * p + 0.25 * f )
                
        test_log_prob = self.gmm.log_prob(test_case)
        
        #print("Test:  ", test_log_prob[0,:].exp())
        #print("Answer:", log_probs[0,:].exp())

        self.assertTrue( torch.all(torch.abs(test_log_prob - log_probs) < 1e-3) )
    """
    def test_sample(self):
        self.gmm.uniform_rate = 0.0
        n = 100000.0
        samps = torch.zeros((self.batch_size, 18), dtype=torch.float32)
        for i in range(int(n)):
            s = self.gmm.sample()
            samps[:,:6] += torch.where(s[:,:6] > 0.5, 1.0, 0.0)
            samps[:,6:] += s[:,6:]
        samps /= n
        
        self.assertTrue(torch.all(torch.abs(samps[:,:6] - 0.75) < 0.01))
        self.assertTrue(torch.all(torch.abs(samps[:,6:12] - 0.0) < 1e-2) )
        self.assertTrue(torch.all(torch.abs(samps[:,12:18] - 1.0) < 1e-2) )
    """
    def test_n_samples(self):
        self.gmm.uniform_rate = 0.0
        n = 100000
        samps = self.gmm.sample(sample_shape=(n,))
        samps[...,:6] = torch.where(samps[...,:6] > 0.5, 1.0, 0.0)
        samps = samps.mean(dim=0)
        print(samps)
        self.assertTrue(torch.all(torch.abs(samps[...,:6] - 0.75) < 0.01))
        self.assertTrue(torch.all(torch.abs(samps[...,6:12] - 0.0) < 1e-2) )
        self.assertTrue(torch.all(torch.abs(samps[...,12:18] - 1.0) < 1e-2) )
    
    def test_entropy(self):
        self.gmm.uniform_rate = 0.0
        entropy = self.gmm.entropy()
        self.assertTrue(entropy.size()[0] == self.batch_size)
    
        
if __name__=="__main__":
    unittest.main()
