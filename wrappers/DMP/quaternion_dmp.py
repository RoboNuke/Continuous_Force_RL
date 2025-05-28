from math import exp
from wrappers.DMP.rbf import RBF
from wrappers.DMP.cs import CS
import torch

class QuaternionDMP():
    def __init__(self, 
                 nRBF:int=100, 
                 betaY: float=1, 
                 dt: float= 0.001, 
                 cs: CS=CS(1.0, 0.001), 
                 num_envs: int=1, 
                 device: str="cpu"
        ):
        self.nRBF = nRBF
        self.ay = 4 * betaY
        self.by = betaY
        self.cs = cs
        self.dt = dt
        self.num_envs = num_envs
        self.device = device

        self.ws = torch.ones((self.num_envs, self.nRBF, 4), device = self.device)
        self.w_out = torch.ones((self.num_envs, self.nRBF, 3), device=self.device)
        self.rbfs = RBF(can_sys=self.cs, num_rbfs=self.nRBF, device=device)

        self.q = torch.zeros((self.num_envs, 1, 4), device = self.device)
        self.n = torch.zeros_like(self.q)
        self.dn = torch.zeros_like(self.n)
        self.gq = torch.zeros_like(self.q)
        self.goal = torch.zeros_like(self.q)
        self.q0 = torch.zeros_like(self.q)
    
    @property
    def w(self):
        self.w_out = self.ws[:,:,1:]
        return self.w_out
    
    def cross(self, a, b):
        # a/b: (num_envs, timestep, 3)
        a1 = a[:,:,0]
        a2 = a[:,:,1]
        a3 = a[:,:,2]
        b1 = b[:,:,0]
        b2 = b[:,:,1]
        b3 = b[:,:,2]

        cros = torch.zeros_like(b)
        cros[:,:,0] = a2 * b3 - a3 * b2
        cros[:,:,1] = a3 * b1 - a1 * b3
        cros[:,:,2] = a1 * b2 - a2 * b1
        return cros


    def quatDiff(self, a, b):
        # quaternion multiplies a * conj(b)
        # a and b: (num_envs, timestep, 4) (w,x,y,z) ordering
        
        conj_b = torch.clone(b)
        conj_b[:,:,1:] *= -1
        #print("Quat diff:", a,conj_b)
        return self.multq(a, conj_b)    

    def multq(self, a, b):
        delta = torch.zeros_like(b)
        v2 = b[:,:,0]
        u2 = b[:,:,1:]
        
        v1 = a[:,:,0]
        u1 = a[:,:,1:]
        #print("Multq inputs")
        #print(v1, v2)
        #print(v2, u2)
        delta[:,:,0] = (v1 * v2) - torch.sum(u1*u2,dim=2)
        delta[:,:,1:] = v1[:,:,None] * u2 + v2[:,:,None] * u1 + self.cross(u1, u2)
        delta /= torch.linalg.norm(delta, dim=2)[:,:,None]
        
        return delta

    def logq(self, a):
        # performs quaternion log on a 
        # a: (num_envs, timestep, 4) (w,x,y,z) q ordering
        # returns (num_envs, timestep, 4) where w is zero always
        
        log_out = torch.zeros_like(a)
        u_norm = torch.linalg.norm(a[:,:,1:], dim=2) 
        log_out[:,:,1:] = torch.acos(a[:,:,0])[:,:,None] * a[:,:,1:] / u_norm[:,:,None]
        log_out[u_norm < 1e-2] = torch.tensor([0,0,0,0], device=self.device, dtype=a.dtype)
        return log_out
    
    def expq(self, a):
        # exp on quaternion a
        # a: (num_envs, timesteps, 4) (w,x,y,z) q ordering
        exp_out = torch.zeros_like(a)
        w_norm = torch.linalg.norm(a, dim=2)
        exp_out[:,:,0] = torch.cos(w_norm)
        #print("expq:", torch.sin(w_norm)[:,:,None].size(), a[:,:,1:].size(), exp_out[:,:,1:].size(), w_norm.size())
        exp_out[:,:,1:] = torch.sin(w_norm)[:,:,None] * a[:,:,1:] / w_norm[:,:,None]
        exp_out[w_norm < 1e-6] = torch.tensor([1,0,0,0], device=self.device, dtype=a.dtype)
        
        return exp_out

    def quatMinus(self, a, b):
        return 2*self.logq(self.quatDiff(a, b))
    
    def learnWeightsCSDT(self, q, n, dn, t):
        # expected input
        # q: quaternion trajectory (num_envs, timestep, 4)
        # n: angular velocity (quaternion with zero complex term) (num_envs, timestep, 4)
        # dn: angular acceleration trajectory (quaternion with zero complex term) (num_envs, timestep, 4)
        # t: time for each above point (timestep)
        #print(self.goal.size(), q.size())
        #return self.learnWeightsCSDT2(q,n,dn,t)
        self.goal[:,:,:] = q[:, -1, :][:,None,:]
        #print("goal:", self.goal)
        x = self.cs.get_xs(t) #self.cs.rollout()
        #print("x:", x)
        self.q0[:,:,:] = q[:, 0, :][:,None,:]
        #print("q0:", self.q0)
        rbMats = self.rbfs.eval(x) #(num_envs, direction, num_rbfs) 
        #print("rbMats:", rbMats)
        fd = dn - self.ay * ( 2 * self.by * self.logq(self.quatDiff(self.goal, q)) - n)
        #print("fd:", fd)
        s = x[None, :, None] * (2 * self.logq(self.quatDiff(self.goal, self.q0)))
        #print("s:", s)
        top = torch.sum(  (s[:,:,None,:] * rbMats[None,:,:,None]) * fd[:,:,None,:], dim=1)
        #print("top:", top)
        top[:,:,0] = 0
        bot = torch.sum( s[:,:,None,:] * rbMats[None,:,:,None] * s[:,:,None,:], dim=1)
        #print("bot:", bot)
        bot += 1e-6
        bot[:,:,0] = 1
        
        self.ws = top / bot

    def learnWeightsCSDT2(self, q, n, dn, t):
        # expected input
        # q: quaternion trajectory (num_envs, timestep, 4)
        # n: angular velocity (quaternion with zero complex term) (num_envs, timestep, 4)
        # dn: angular acceleration trajectory (quaternion with zero complex term) (num_envs, timestep, 4)
        # t: time for each above point (timestep)
        #print(self.goal.size(), q.size())
        self.goal[:,:,:] = q[:, -1, :][:,None,:]
        x = self.cs.rollout()
        
        self.q0[:,:,:] = q[:, 0, :][:,None,:]
        rbMats = self.rbfs.eval(x) #(num_envs, direction, num_rbfs) 
        
        gq0_dist = self.quatMinus(self.goal, self.q0)#(2 * self.logq(self.quatDiff(self.goal, self.q0)))

        a = (dn/self.ay) + n
        print(a.size())
        print(gq0_dist.size(), x[None,:,None].size(), self.quatMinus(self.goal, q).size())
        b = self.quatMinus(self.goal, q) - gq0_dist * x[None,:,None]   #2 * self.logq(self.quatDiff(self.goal, q))
        print(b.size())
        fd = a/self.by - b
        print(fd[0,:5,1])


        #fd = dn - self.ay * ( 2 * self.by * self.logq(self.quatDiff(self.goal, q)) - n)
        
        s = x[None, :, None] #* gq0_dist
        
        print(s.size(), rbMats.size(), fd.size())
        print(s[:,:,None,:].size(), rbMats[None,:,:,None].size(), fd[:,:,None,:].size())
        top = torch.sum(  (s[:,:,None,:] * rbMats[None,:,:,None]) * fd[:,:,None,:], dim=1)
        top[:,:,0] = 0
        bot = torch.sum( s[:,:,None,:] * rbMats[None,:,:,None] * s[:,:,None,:], dim=1)
        bot[:,:,0] = 1
        
        self.ws = top / bot
        print(self.w)
            
        

    def calcWPsi(self, x: float):
        thee = self.rbfs.eval(x)
        
        top =  torch.sum(thee[None, :, None] * self.ws, dim=1)[:,None,:]
        
        bot = torch.sum(thee)
        return top / bot
    
    def step(self, tau=1.0, error=0.0, external_force=None):
        ec = 1.0 / (1.0+error)

        x = self.cs.step(tau=tau, error_coupling = ec)
        
        F = self.quatMinus(self.goal, self.q0) * self.calcWPsi(x) * x 
        #F = self.calcWPsi(x) * x
        
        self.dn = self.ay * ( 2*self.by * self.logq(self.quatDiff(self.goal, self.q)) - self.n) + F
        #self.dn = self.ay * ( self.by * (self.quatMinus(self.goal, self.q) - self.quatMinus(self.goal, self.q0) * x + F) - self.n)
        #self.ddy = self.ay * (self.by * (self.goal - self.y - (self.goal - self.y0) * x + F) - self.dy)

        self.n += self.dn * self.dt * ec / tau
        self.q = self.multq( self.expq(self.n * self.dt / (2 * tau)), self.q)
        
        return self.q, self.n, self.dn

    def reset(self, goal, q: torch.tensor, n = None, dn = None):
        self.q[:,:,:] = q
        if n is None:
            self.n = torch.zeros_like(self.q)
        else:
            self.n = n
        if dn is None:
            self.dn = torch.zeros_like(self.q)
        else:
            self.dn = dn
            #print(self.dn.size(), dn.size())
        self.q0 = torch.clone(self.q)
        self.goal[:,:,:] = goal
        self.cs.reset()

    def rollout(self, g, q0, n0=None, dn0=None, tau=1, scale=1):
        self.reset(g, q0, n0, dn0)
        t = 0.0
        timesteps = int(self.cs.timesteps * tau)
        z = torch.zeros((self.num_envs, timesteps+1, 4), device = self.device)
        w = torch.zeros_like(z)
        dw = torch.zeros_like(w)
        ts = torch.zeros(timesteps+1, device = self.device)
        
        z[:,0,:][:,None,:] = self.q
        w[:,0,:][:,None,:] = self.n
        dw[:,0,:][:,None,:] = self.dn
        ts[0] = t

        for it in range(timesteps):
            #t = np.log(self.cs.x) / -self.cs.ax
            t += self.cs.dt
            #print(self.q)
            self.step(tau=tau, error=0.0, external_force=None)
            #print(self.q)
            #assert 1 == 0
            #print(self.q, "\t", torch.linalg.norm(self.q))
            z[:,it+1, :][:,None,:] = self.q
            w[:,it+1, :][:,None,:] = self.n
            dw[:,it+1, :][:,None,:] = self.dn
            ts[it+1] = t

        return(ts, z, w, dw)

def pltqndn(fig, axs, t, data_ref, num_envs=1, names=['x', 'y', 'z', 'w'], colors=['r-','g-' ,'b-','m-']):
    
    for env_id in range(num_envs):
        for j in range(len(names)):
            for i in range(len(data_ref)):
                idx = j
                if 'w' in names[j] and i > 0:
                    continue
                if i == 0 and 'w' not in names[j]:
                    idx+=1
                elif i == 0 and 'w' in names[j]:
                    idx = 0
                #if names[j] in ['x','y','z']:
                #    continue
                #print(names[j], colors[j])
                if env_id > 1:
                    axs[i, env_id].plot(t, data_ref[i][env_id,:,idx], color=colors[j][0], linestyle=colors[j][1])
                else:
                    axs[i].plot(t, data_ref[i][env_id,:,idx], color=colors[j][0], linestyle=colors[j][1], label=names[j])
                #if colors[j][0] == 'k':
                #    plt.show()
    #fig.legend(names)


    return fig, axs


if __name__=="__main__":

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.spatial.transform import Rotation

    dt = 1/500
    tmax = 1.0
    num_envs = 2
    dmp = QuaternionDMP(
        nRBF=10, 
        betaY=6/4, 
        dt=dt, 
        cs=CS(ax=2, dt=dt), 
        num_envs=num_envs
    )

    t = torch.linspace(start=0.0, end=tmax, steps= int(tmax/dt)+1 ) #/ tmax
    
    of = 1.0
    ns = 0.1
    beta = torch.tensor(3.14159/2)
    omega = torch.tensor(5.0)

    q = torch.zeros(num_envs, len(t), 4)
    n = torch.zeros(num_envs, len(t), 4)
    dn = torch.zeros_like(n)


    """
    q[:,:,0] = torch.cos(beta/2.0)
    q[:,:,1] = torch.sin(beta/2.0) * torch.cos(omega * t) 
    q[:,:,2] = torch.sin(beta/2.0) * torch.sin(omega * t) 
    q[:,:,3] = 0.0

    n[:,:,0] = -omega * torch.sin(beta) * torch.sin(omega * t)  #/ (omega**2 + 1 - 2 * torch.cos(beta))
    n[:,:,1] = omega * torch.sin(beta) * torch.cos(omega * t) #/ (omega**2 + 1 - 2 * torch.cos(beta))
    n[:,:,2] = omega * (torch.cos(beta)-1) #/ (omega**2 + 1 - 2 * torch.cos(beta))
    

    dn[:,:,0] = - (omega**2) * torch.sin(beta) * torch.cos(omega * t) #/ (omega**2 + 1 - 2 * torch.cos(beta))
    dn[:,:,1] = - (omega**2) * torch.sin(beta) * torch.sin(omega * t) #/ (omega**2 + 1 - 2 * torch.cos(beta))
    dn[:,:,2] = 0.0
    #print(n[:,:,0])
    #assert torch.all(n[:,:,0] < 1e-6), "Multiplaction not correct"
    #assert torch.all(dn[:,:,0] < 1e-6), 'accel mujlt not correct'
    #t *= tmax
    """


    a = 1
    b = -2
    c = 3
    d = -4
    scaling_factor = 10
    for i in range(num_envs):
        q[i,0,0] = 1 # we start at unit quaternion
        #print("Init t:", q)
        # define the angular velocity
        n[i:,:,0] = 0.0 
        n[i,:,1] = scaling_factor*(torch.sin(a * t) + torch.cos(b * t))
        n[i,:,2] = scaling_factor * (torch.sin(c * t) + torch.cos(d * t))
        n[i,:,3] = 0.0

        # define angular acceleration
        dn[i,:,0] = 0.0
        dn[i,:,1] = scaling_factor*a * torch.cos(a * t) - scaling_factor * b * torch.sin(b * t)
        dn[i,:,2] = scaling_factor * (c * torch.cos(c * t) - d * torch.sin(d * t))
        dn[i,:,3] = 0

    # numerical integration to get the quaternion positions
        dt = t[1] - t[0]
        for k, ti in enumerate(t[1:]):
            #print("i:", i, ti)
            #print("\t1) ", n[:,i,:][:,None,:] * dt / 2)
            #print("\t2) ", dmp.expq(n[:,i,:][:,None,:] * dt / 2), q[:,i-1,:][:,None,:])
            q[i,k+1,:] = dmp.multq( dmp.expq(n[i,k,:][None,None,:] * dt / 2), q[i,k,:][None,None,:])
            #print("\t3) ", q[:,i,:])


    fig, axs = plt.subplots(3, 1)
    fig.set_figwidth(800/96)
    fig.set_figheight(1000/96)
    fig.tight_layout(pad=5.0)
    fig, axs = pltqndn(fig, axs, t, data_ref=[q, n, dn])
    #plt.show()

    #dmp.reset(q[:,-1,:], q, n, dn)
    dmp.learnWeightsCSDT(q,n,dn,t)
    
    tau = 1.0
    scale =1
    g = q[:,-1,:] * scale
    
    ts, z, w, dw = dmp.rollout(g[:,None,:], q[:,0,:][:,None,:], n[:,0,:][:,None,:], dn[:,0,:][:,None,:], tau, scale)
    
    #print(z.size(), w.size(), dw.size())
    #for i in [0,1,2,3]:
    #    fig = plt.figure(2)
    #    plt.plot(ts, z[0,:,i], 'k')
    #    plt.plot(t, q[0,:,i], 'r')
    #    plt.show()
    fig, axs = pltqndn(fig, axs, ts, data_ref=[z, w, dw],
                       names=["x - Learned", "y - Learned", "z - Learned", "w - Learned"],
                        #colors=['r:','g:' ,'b:','m:']) 
                        colors=['k:' for i in range(4)]) #
    handles, labels = axs[0].get_legend_handles_labels()
    #labels[4] = 'Est'
    #fig.legend(handles[:5], labels[:5])
    #plotSub(axs[0], t, ts, y, z,f"Position DMP ({dt})", "Position")
    #axs[0].plot(t, y1, 'g')
    #plotSub(axs[1], t, ts, dy, dz, "Velocity DMP", "Velocity")
    #axs[1].plot(t, dy1, 'g')
    #plotSub(axs[2], t, ts, ddy, ddz, "Accel DMP", "Acceleration")
    #axs[2].plot(t, ddy1, 'g')
    for axis in axs:
        axis.set(xlabel = "time (s)")
    axs[0].set_title("Quaternion")
    axs[0].set(ylabel="Component Value (rad)")
    axs[1].set_title("Angular Velocity")
    axs[1].set(ylabel="Component Value (rad/sec)")
    axs[2].set_title("Angular Acceleration")
    axs[2].set(ylabel="Component Value (rad/sec^2)")

    plt.legend()
    fig.suptitle("Quaternion DMPs fixed <1 day of debugging", fontsize=20)
    plt.show()
    """
    for j in range(10):
        g = 25 * np.random.rand()
        for i in range(10):
            dmp.ws = np.random.normal(0, 100, dmp.ws.shape)
            #dmp.reset(0, 0, 0)
            ts, z, dz, ddz = dmp.rollout(g=g, y0=0)
            plt.plot(ts, z)
        #plt.show()
    """