import torch
import matplotlib.pyplot as plt

class CS():
    def __init__(self, ax:float = 5.0, dt:float=0.01, device="cpu"): #, num_envs: int = 1):
        self.dt = dt
        self.timesteps = int(1.0 / self.dt)
        self.ax = ax
        self.device = device
        #self.num_envs = num_envs

        self.x = 1.0 # torch.ones((num_envs,), device=self.device)
        self.xPath = torch.zeros((self.timesteps), device=self.device)

    def reset(self):
        self.x = 1.0

    def step(self, tau=1.0, error_coupling=1.0):
        self.x += (-self.ax * self.x * error_coupling) * self.dt / tau
        if self.x < 0.0001:
            self.x = 0.00005
        return self.x

    def rollout(self, tau=1.0):
        self.reset()
        self.xPath *= 0.0
        self.xPath = torch.zeros((int(self.timesteps * tau)+1), device = self.device)
        for t in range( int(self.timesteps * tau)):
            self.xPath[t] = self.x
            self.step(tau=tau)
        return self.xPath

    def get_xs(self, t):
        return torch.exp(-self.ax * t)
    
if __name__=="__main__":
    def plt_cs(cs, path, tau=1.0):
        t = np.linspace(0,1,len(path)) * tau
        plt.figure(1)
        plt.plot(t, path[:])

    import numpy as np
    cs = CS(5.0, 0.01)#,num_envs=3)
    path = cs.rollout()
    plt_cs(cs, path)
    #plt.show()

    # test error coupling
    # test tau
    #path = cs.rollout(tau=2.0)
    #plt_cs(cs, path, tau=2.0)
    #plt.show()

    plt.figure(3)
    taus = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    for tau in taus:
        path = cs.rollout(tau=tau)
        t = np.linspace(0,1,len(path)) * tau
        plt.plot(t, path)
    plt.legend(taus)
    plt.show()
