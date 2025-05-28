from math import exp
import torch
import matplotlib.pyplot as plt

from wrappers.DMP.cs import CS

class RBF():
    def __init__(self, can_sys: CS, num_rbfs: int=8, device: str='cpu'):
        self.device = device
        self.num_rbfs = num_rbfs
        self.can_sys = can_sys
        self.centerRBFsV2()


    def eval(self, x):
        # x should be [timesteps] size
        if type(x) == float:
            dx = x - self.cs
        else:
            #print(x.device, self.cs.device)
            dx = x[:,None] - self.cs[None, :]
        return torch.exp(-self.hs * (dx *dx))
    
    #def theeMat(self, x):
    #    mat = self.eval(x)
    #    return mat
    
    def centerRBFs(self):
        des_c = torch.linspace(0, 1.0, self.num_rbfs, device=self.device)#+2)[1:-1]
        self.cs = torch.exp(-self.can_sys.ax * des_c)
        self.hs = self.num_rbfs / (self.cs ** 2) 




    def centerRBFsV2(self):
        self.cs = torch.exp(-self.can_sys.ax * torch.linspace(0,1, self.num_rbfs, device=self.device))
        #print("cs:", self.cs)
        self.hs = torch.zeros_like(self.cs)
        for i in range(self.num_rbfs -1):
            self.hs[i] = 1 / ((self.cs[i+1] - self.cs[i])**2)
        self.hs[-1] = self.hs[-2]
        #print(self.cs)
        #print("hs:", self.hs)
    """
    def centerRBFsV2(self):
        self.RBFs = []
        c = [np.exp(-self.cs.ax * i/self.nRBF) for i in range(self.nRBF)]
        h = [1 / ((c[i+1] - c[i])**2) for i in range(self.nRBF-1)]
        h.append(h[-1])
        #print(c)
        #print(h)
        for i in range(self.nRBF):
            self.RBFs.append(RBF(c[i],h[i]))
    """
    
if __name__=="__main__":
    import numpy as np
    #num_envs = 3
    num_rbf = 5
    dt = 1/200.0 #0.0001
    """ax = 5.0

    cs = CS(dt=dt)

    rbfs = RBF(cs, num_rbf)

    xs = cs.rollout()
    print(xs.size())
    ts = np.linspace(0, 1.0, int(1/dt))
    print(ts.shape)
    ys = rbfs.eval(xs)
    print(ys[:,:].size())
    plt.figure(1)
    plt.plot(ts, ys[:,:])
    plt.title("RBFs as function of Time")
    plt.figure(2)
    plt.plot(xs[:], ys[:,:])
    plt.title("RBFs as function of CS")
    plt.gca().invert_xaxis()
    plt.show()
    """
    ax_over_list = [1.0, 2.5, 5, 7.5, 10]
    color = ['r', 'b', 'g']
    for ax_item in ax_over_list:
        ax_list = [ax_item]
        plt.figure(1)
        for ax in ax_list:
            cs = CS(ax=ax, dt=dt)
            rbfs = RBF(cs, num_rbf)
            xs = cs.rollout()
            ts = np.linspace(0, 1.0, int(1/dt))
            ys = rbfs.eval(xs)
            plt.plot(ts, ys[:,:])#, color=color[ax_over_list.index(ax)])
        plt.title("RBFs as function of Time")
        plt.legend(ax_list)
        plt.figure(2)
        for ax in ax_list:
            cs = CS(ax=ax, dt=dt)
            rbfs = RBF(cs, num_rbf)
            xs = cs.rollout()
            ts = np.linspace(0, 1.0, int(1/dt))
            ys = rbfs.eval(xs)
            plt.plot(xs[:], ys[:,:])#, color=color[ax_over_list.index(ax)])
        plt.title("RBFs as function of CS")
        plt.gca().invert_xaxis()
        plt.legend(ax_list)

        plt.show()

    