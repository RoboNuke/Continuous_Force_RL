from math import exp
import matplotlib.pyplot as plt
import numpy as np
from wrappers.DMP.rbf import RBF
from wrappers.DMP.cs import CS
import torch

class DiscreteDMP():
    def __init__(self, 
                 nRBF:int=100, 
                 betaY: float=1, 
                 dt: float= 0.001, 
                 cs: CS=CS(1.0, 0.001), 
                 num_envs: int=1, 
                 num_dims: int=1,
                 device: str="cpu"
        ):
        self.nRBF = nRBF
        self.ay = 4 * betaY
        self.by = betaY
        self.cs = cs
        self.dt = dt
        self.num_envs = num_envs
        self.num_dims = num_dims
        self.device = device

        self.ws = torch.ones((self.num_envs, self.nRBF, self.num_dims), device = self.device) #[1.0 for x in range(self.nRBF)]

        self.rbfs = RBF(can_sys=self.cs, num_rbfs=self.nRBF, device=self.device)

        self.ddy = 0
        self.dy = 0
        self.y = 0
        self.goal = None
        self.y0 = None


    def learnWeightsCSDT(self, y, dy, ddy, t):
        # expected input
        # y: position trajectory (num_envs, timestep, num_dims )
        # dy: velocity trajectory (num_envs, timestep, num_dims )
        # ddy: acceleration trajectory (num_envs, timestep, num_dims )
        # t: time for each above point (timestep)
        # points are assumed to be pl

        self.goal = y[:,-1,:].clone()
        x = self.cs.get_xs(t) #self.cs.rollout()
        
        self.y0 = y[:,0,:].clone()
        
        rbMats = self.rbfs.eval(x) 
        
        a = (ddy/self.ay) + dy
        
        b = (self.goal - self.y0)[:,None,:] * x[None,:,None] + y - self.goal[:,None,:]
        
        fd = a/self.by + b

        #print("fd:", fd)
        s = x[None,:,None] #* (self.goal - self.y0)[:,None,:]
        #print(x[None,:,None].size(), s.size())
        
        #print(rbMats.size())
        #print(s[:,:,None,:].size(), rbMats[None,:,:,None].size())
        #print( (s[:,:,None,:] * rbMats[None,:,:,None]).size(), fd[:,:,None,:].size())
        #top = torch.sum(  (s[:,None] * rbMats)[None,:,:,None] * fd[:,:,None,:], dim=1)
        top = torch.sum(  (s[:,:,None,:] * rbMats[None,:,:,None]) * fd[:,:,None,:], dim=1)
        #bot = torch.sum( s[:,None] * rbMats * s[:,None], dim=0)
        bot = torch.sum( s[:,:,None,:] * rbMats[None,:,:,None] * s[:,:,None,:], dim=1)
        bot += 1e-6
        #print(top.size(), bot.size())
        #self.ws = top / bot[None,:,None]
        #print(self.ws.size())
        self.ws = top / bot

        #print(self.ws.size())

        #delta = (self.goal[:,None,:]  - self.y0[:,None,:])
        #delta = delta.repeat( (1,self.nRBF,1))
        #delta_idx = torch.logical_or(delta > 0.0001, delta < 0.0001)
        #self.ws[delta_idx] /= delta[delta_idx] 
        #self.ws /= delta

    def learnWeights(self,y, ydot, ydotdot, tt):
        self.goal = y[-1]
        x = self.cs.rollout()
        #dt = t / t[-1]
        path = torch.zeros((len(x)), device=self.device)
        dy = torch.zeros((len(x)), device=self.device)
        ddy = torch.zeros((len(x)), device=self.device)
        ts = torch.zeros(len(x), device=self.device)
        #t=torch.linspace(0, 1.0, len(y))
        
        
        # TODO Torch Interpolate
        import scipy.interpolate
        path_gen = scipy.interpolate.interp1d(tt, y)
        dy_gen = scipy.interpolate.interp1d(tt, ydot)
        ddy_gen = scipy.interpolate.interp1d(tt, ydotdot)
        import math
        for i in range(len(x)):
            #tx = min(1*np.log(x[i]) / -self.cs.ax, 1.0)
            tx = i * self.cs.dt
            print(tx)
            #$print(x[i], tx, i, i/len(x))
            path[i] = float(path_gen(tx))
            dy[i] = float(dy_gen(tx))
            ddy[i] = float(ddy_gen(tx))
            ts[i] = tx #i * self.dt
        #assert 1 ==0
        # estimate the gradients
        #ddt = self.dt * t[-1]
        # TODO Torch gradients
        #dy = np.gradient(path)#/(self.dt)
        #ddy = np.gradient(dy)#/(self.dt)

        #dy = torch.from_numpy(dy)
        
        #ddy = torch.from_numpy(ddy)
        """
        print(min(tt), max(tt), min(ts), max(ts))
        plt.plot(ts, path, '-r')
        plt.plot(tt/tt[-1], y)
        plt.show()
        plt.plot(ts ,dy, '-r')
        plt.plot(tt/tt[-1], ydot)
        plt.show()
        plt.plot(ts,ddy, '-r')
        plt.plot(tt/tt[-1], ydotdot)
        plt.show()
        """
        y = path
        
        self.y0 = y[0]
        #x = x.unsqueeze(1)
        # rbMats = (len(x), num_rbfs)
        rbMats = self.rbfs.eval(x) #[self.RBFs[i].theeMat(x) for i in range(self.nRBF)]

        fd = ddy - self.ay * (self.by * (self.goal - y) - dy)
        
        s = x #* (self.goal - self.y0)
        top = torch.sum(s[:,None] * rbMats * fd[:,None], dim=0)
        bot = torch.sum( s[:,None] * rbMats * s[:,None], dim=0)

        self.ws = top / bot

        if abs(self.goal  - y[0]) > 0.0001:
            self.ws /= (self.goal - y[0])

            

    def calcWPsi(self, x: float):
        thee = self.rbfs.eval(x) 
        #print(thee.size(), self.ws.size() )
        top =  torch.sum(thee[None,:,None] * self.ws, dim=1)
        bot = torch.sum(thee, dim=0)
        

        bot[bot <= 1e-6] = 1.0
        #print(bot)
        #print("calcWPsi tb:", top.size(), bot.size())
        #print( (top/bot).size())
        return top / bot
    
    def step(self, tau=1.0, error=0.0, external_force=None):
        ec = 1.0 / (1.0+error)

        x = self.cs.step(tau=tau, error_coupling = ec)
        
        #print(self.calcWPsi(x).size(), self.goal.size(), self.y0.size())
        F = self.calcWPsi(x) * x #(self.goal - self.y0) * x

        #self.ddy = self.ay * (self.by * (self.goal - self.y) - self.dy) + F
        self.ddy = self.ay * (self.by * (self.goal - self.y - (self.goal - self.y0) * x + F) - self.dy)

        if external_force != None:
            self.ddy += external_force
        
        self.dy += self.ddy * self.dt * ec / tau
        self.y += self.dy * self.dt * ec / tau

        #print(self.y.size(), self.dy.size(), self.ddy.size())

        return self.y, self.dy, self.ddy

    def reset(self, goal, y, dy = 0.0, ddy = 0.0):
        self.y = y.clone()
        self.dy = dy.clone()
        self.ddy = ddy.clone()
        self.y0 = self.y.clone()
        self.goal = goal.clone()
        self.cs.reset()

    def rollout(self, g, y0, dy0=0, ddy0=0, tau=1, scale=1):
        self.reset(g, y0, dy0, ddy0)
        t = 0.0

        # allocate memory
        timesteps = int(self.cs.timesteps * tau)
        z = torch.zeros( (self.num_envs, timesteps+1, self.num_dims), device=self.device)
        dz = torch.zeros_like(z)
        ddz = torch.zeros_like(dz)
        ts = torch.zeros(timesteps+1)
        #print(self.dt)
        ts[0] = t
        z[:,0,:] = self.y
        dz[:,0,:] = self.dy
        ddz[:,0,:] = self.ddy
        #print(f"Total Time:{self.cs.run_time * tau}")
        for it in range(timesteps):
            #t = np.log(self.cs.x) / -self.cs.ax
            t += self.cs.dt

            self.step(tau=tau, error=0.0, external_force=None)

            z[:,it+1,:] = self.y
            dz[:,it+1,:] = self.dy
            ddz[:,it+1,:] = self.ddy
            ts[it+1] = t

        return(ts, z, dz, ddz)


def plotSub(ax, t, ts, org, cpy, tit="DMP", ylab="Function"):
    ax.plot(t, org)
    ax.plot(ts, cpy,'r--')
    #ax.plot(np.linspace(0, t[-1], len(cpy)), cpy,'r--')
    ax.set(ylabel=ylab)
    ax.set_title(tit,  fontsize=20)

def single_dim_test():
    dt = 10.0/250.0
    for dt in [1/200.0, 10/200.0, 10/500, 1/500]:
        
        tmax = 1.0
        
        dmp = DiscreteDMP(nRBF=10, betaY=12.5/4.0, dt=dt, cs=CS(ax=2.5, dt=dt))

        t = np.arange(0, tmax, dt) / tmax
        
        of = 1.0
        ns = 0.1
        y1 = np.sin(of* 10*t) + np.cos(of * 3 *t) 
        y = y1 + np.random.normal(0, ns, t.shape)
        dy1 = of * 10*np.cos(of* 10*t) - of*3*np.sin(of*3*t) 
        dy = dy1 + np.random.normal(0, ns * 10, t.shape)
        ddy1 = -100* of**2 * np.sin(of*10*t) - 9 * of**2 * np.cos(of*3*t) 
        ddy = ddy1 + np.random.normal(0, ns * 100, t.shape)
        t *= tmax

        print("single:", y.shape, dy.shape, ddy.shape)
        dmp.learnWeightsCSDT(y,dy,ddy,torch.from_numpy(t))
        print("Weights:", dmp.ws)
        return
        tau = 1.0
        scale =1
        g = y[-1] * scale
        
        dmp.reset(y[0], dy[0], ddy[0])
        ts, z, dz, ddz = dmp.rollout(g, y[0], dy[0], ddy[0], tau, scale)
        
        fig, axs = plt.subplots(3)
        fig.set_figwidth(800/96)
        fig.set_figheight(1000/96)
        fig.tight_layout(pad=5.0)
        print(t[-1], ts[-1] )
        plotSub(axs[0], t, ts, y, z,f"Position DMP ({dt})", "Position")
        axs[0].plot(t, y1, 'g')
        plotSub(axs[1], t, ts, dy, dz, "Velocity DMP", "Velocity")
        axs[1].plot(t, dy1, 'g')
        plotSub(axs[2], t, ts, ddy, ddz, "Accel DMP", "Acceleration")
        axs[2].plot(t, ddy1, 'g')
        plt.xlabel("time (s)")

        plt.legend(['Noisy Function', 'Learned Trajectory', 'Original Function'])
    #plt.show()

    for j in range(10):
        g = np.random.rand()
        print("Goal:", g)
        for i in range(10):
            dmp.ws = np.random.normal(0, 10, dmp.ws.shape)
            dmp.reset(0, 0, 0)
            ts, z, dz, ddz = dmp.rollout(g=g, y0=0)
            plt.plot(ts, z)
        plt.show()


def test(num_dims = 1, num_envs = 1, noise = 0.1,
         tau=1.0, scale=1.0, tmax=1.0, dt=1/50, fp="/home/hunter/Pictures/profiles"):
    # define 3-D trajectories
    dmp = DiscreteDMP(
        nRBF=10, 
        betaY=12.5/4.0, 
        dt=dt, 
        cs=CS(ax=2.5, dt=dt/tmax), 
        num_envs=num_envs, 
        num_dims=num_dims
    )

    t = torch.linspace(start=0, end=tmax, steps= int(tmax/dt)+1 ) 
    
    sco = torch.tensor([10.0, 3, 5])
    cco = torch.tensor([3,  6, 1.5])
    
    cco = cco.repeat(num_envs, 1)
    sco = sco.repeat(num_envs,1)
    for i in range(num_envs):
        if i > 2:
            cco[i,:] *= 0.25
            sco[i,:] *= 0.25
        else:
            cco[i,:] = torch.roll(cco[i,:], i)
    if num_dims < 3:
        sco = sco[:,:num_dims]
        cco = cco[:,:num_dims]
        
    y = torch.sin(sco[:,None,:] * t[None,:,None]) + torch.cos(cco[:,None,:] * t[None,:,None]) 
    y[0,:,0] = 3
    y[-1,:,-1] = 3 # make an example with a small change in value
    diff = 0.01
    y[-1,-1,-1] += diff
    for i in range(int(tmax/dt)+1):
        y[-1, i, -1] = y[-1,0,-1] + diff * i / (int(tmax/dt)+1)

    dy = sco[:,None,:] * torch.cos(sco[:,None,:]*t[None,:,None]) - cco[:, None,:] * torch.sin(cco[:, None,:]*t[None, :,None])
    dy[-1,:,-1] = 0
    dy[0,:,0] = 0
    ddy = -sco[:, None,:]**2 * torch.sin(sco[:,None,:]*t[None,:,None]) - cco[:,None,:]**2 * torch.cos(cco[:,None,:]*t[None:,None])
    ddy[-1,:,-1] = 0
    ddy[0,:,0] = 0
    #ddy *= 0
    #print(y.size(), dy.size(), ddy.size())

    fig, axs = plt.subplots(num_envs, num_dims)
    fig.set_figwidth(num_dims / 3 * 1600/96)
    fig.set_figheight(num_envs / 4 * 1000/96)
    fig.tight_layout(pad=5.0)

    dmp.learnWeightsCSDT(y, dy, ddy, t)
    #print(torch.max(dmp.ws), torch.min(dmp.ws), torch.mean(dmp.ws))
    #dmp.reset(y[:,0,:], dy[:,0,:], ddy[:,0,:])
    #print("goal:", y[:,-1,i])
    ts, z, dz, ddz = dmp.rollout(y[:,-1,:], y[:,0,:], dy[:,0,:], ddy[:,0,:], tau, scale)
    ts *= tmax
    #z = 0.95 * y
    if num_dims > 1 and num_envs > 1:
        for i in range(num_envs):
            for j in range(num_dims):
                axs[i,j].plot(t, y[i,:,j], label="Original")
                axs[i,j].plot(ts, z[i,:,j], 'r--', label="Fit DMP")
                axs[i,j].set_title(f"DMP (env={i}, dim={j})",  fontsize=20)
                axs[i,j].set(xlabel = 'Time (s)')
                axs[i,j].set(ylabel ='Position (m)')
    elif num_dims > 1 and num_envs == 1:
        for j in range(num_dims):
            axs[j].plot(t, y[0,:,j], label="Original")
            axs[j].plot(ts, z[0,:,j], 'r--', label="Fit DMP")
            axs[j].set_title(f"DMP (dim={j})",  fontsize=20)
            axs[j].set(xlabel = 'Time (s)')
            axs[j].set(ylabel ='Position (m)')
    elif num_dims == 1 and num_envs > 1:
        for i in range(num_envs):
            axs[i].plot(t, y[i,:,0], label="Original")
            axs[i].plot(ts, z[i,:,0], 'r--', label="Fit DMP")
            axs[i].set_title(f"DMP (env={i})",  fontsize=20)
            axs[i].set(xlabel = 'Time (s)')
            axs[i].set(ylabel ='Position (m)')
    else:
        axs.plot(t, y[0,:,0], label="Original")
        axs.plot(ts, z[0,:,0], 'r--', label="Fit DMP")
        axs.set_title(f"DMP",  fontsize=20)
        axs.set(xlabel = 'Time (s)')
        axs.set(ylabel ='Position (m)')

    plt.legend()
    #plt.savefig(f"{fp}/fit_performance_{int(tmax/dt)}.png")
    plt.show()


def data_sensitivity_test(plot=True, fp="/home/hunter/Pictures/profiles", max_t=101, nRBFs=10):
    # straight line, small increase, s-curve, y0=g curve, up curve, down curve
    # define 3-D trajectories
    sco = torch.tensor([[0,10.0, 5.0],[0, 2.5, 5]])
    cco = torch.tensor([[0,1.5, 1.5],[0, 0.75, 6]])

    names = [["Constant", "S-Curve", "Init = Goal Curve"],["Small Increase", "Term Up Curve", "Term Down Curve"]]

    tmax = 0.5
    tau = 1.0
    scale = 1.0
    noise = 0.1
    num_envs = 2
    num_dims = 3
    t_vals = [k * 10 for k in range(2,max_t)]
    dts = [1/(10*k-1) for k in range(2,max_t)]
    
    error = torch.zeros((len(dts), num_envs, num_dims))
    for dt_idx, dt in enumerate(dts):
        dmp = DiscreteDMP(
            nRBF=nRBFs, 
            betaY=25/4.0, 
            dt=dt, 
            cs=CS(ax=5, dt=dt/tmax), 
            num_envs=num_envs, 
            num_dims=num_dims
        )
        #print("cs dt:", dt/tmax)
        #print(dmp.cs.dt, dmp.cs.timesteps)
        #print("dt", dt, int(tmax/dt))

        t = torch.linspace(start=0, end=tmax, steps= int(tmax/dt)+1 ) 
        y = torch.sin(sco[:,None,:] * t[None,:,None]) + torch.cos(cco[:,None,:] * t[None,:,None]) 
        
        y[0,:,0] = 3
        y[1,:,0] = 3 # make an example with a small change in value
        diff = 0.01
        y[1,-1,0] += diff
        for i in range(int(tmax/dt)+1):
            y[1,i,0] = y[1,0,0] + diff * i / (int(tmax/dt)+1)

        dy = sco[:,None,:] * torch.cos(sco[:,None,:]*t[None,:,None]) - cco[:, None,:] * torch.sin(cco[:, None,:]*t[None, :,None])
        dy[1,:,0] = 0
        dy[0,:,0] = 0
        ddy = -sco[:, None,:]**2 * torch.sin(sco[:,None,:]*t[None,:,None]) - cco[:,None,:]**2 * torch.cos(cco[:,None,:]*t[None:,None])
        ddy[1,:,0] = 0
        ddy[0,:,0] = 0

        dmp.learnWeightsCSDT(y, dy, ddy, t)
        #print(torch.max(dmp.ws), torch.min(dmp.ws), torch.mean(dmp.ws))
        #dmp.reset(y[:,0,:], dy[:,0,:], ddy[:,0,:])
        #print("goal:", y[:,-1,i])
        ts, z, dz, ddz = dmp.rollout(y[:,-1,:], y[:,0,:], dy[:,0,:], ddy[:,0,:], tau, scale)
        ts *= tmax
        
        error[dt_idx,:,:] = torch.sum((y - z) * (y - z), 1) / t_vals[dt_idx] # sum squred error
        #error[dt_idx,0, 2] *= 0
        
        if plot:
            if t_vals[dt_idx] in [20, 30, 50, 200, 500, t_vals[-1]]:
                fig, axs = plt.subplots(num_envs, num_dims)
                fig.set_figwidth(num_dims / 3 * 1600/96)
                fig.set_figheight(num_envs / 4 * 1000/96)
                fig.tight_layout(pad=5.0)
                for i in range(num_envs):
                    for j in range(num_dims):
                        axs[i,j].plot(t, y[i,:,j], label="Original")
                        axs[i,j].plot(ts, z[i,:,j], 'r--', label="Fit DMP")
                        axs[i,j].set_title(names[i][j])
                        axs[i,j].set(xlabel = '# Data Points')
                        axs[i,j].set(ylabel ='Sum Squared Error')
                fig.suptitle(f"Fit at {t_vals[dt_idx]} Data Points",  fontsize=20)
                plt.legend()
                plt.savefig(f"{fp}/{t_vals[dt_idx]}_fit.png")
                #plt.show()

    if plot:   
        # plot type error + average error
        fig, axs = plt.subplots(num_envs, num_dims)
        fig.set_figwidth(num_dims / 3 * 1600/96)
        fig.set_figheight(num_envs / 4 * 1000/96)
        fig.tight_layout(pad=5.0)
        
        for dim_idx in range(num_dims):
            for env_idx in range(num_envs):
                axs[env_idx,dim_idx].plot(t_vals, error[:,env_idx, dim_idx])
                axs[env_idx, dim_idx].set_title(names[env_idx][dim_idx],  fontsize=20)
                axs[env_idx, dim_idx].set(xlabel = '# Data Points')
                axs[env_idx, dim_idx].set(ylabel ='Sum Squared Error')
        plt.savefig(f"{fp}/individual_error.png")
    
    plot_set = [50]

    tot_error = [torch.sum(error[i,:,:]) for i in range(len(t_vals))]
    if plot:
        fig2, ax2 = plt.subplots()
        ax2.plot(t_vals, tot_error)
        ax2.set_title("Total Error")
        ax2.set(xlabel = "# Data Points")
        ax2.set(ylabel = "Sum Squared Error")
        for val in plot_set:
            ax2.plot([val,val],[0,max(tot_error)], 'k--', label=str(val) + " Data Pts")
        ax2.legend()
        plt.savefig(f"{fp}/total_error.png")
        #plt.show()

        fig3, ax3 = plt.subplots()
        for dim_idx in range(num_dims):
            for env_idx in range(num_envs):
                ax3.plot(t_vals, error[:,env_idx, dim_idx], label=names[env_idx][dim_idx])
        ax3.set_title("Individual Error")
        ax3.set(xlabel = "# Data Points")
        ax3.set(ylabel = "Sum Squared Error")
        for val in plot_set:
            ax3.plot([val,val],[0,max(tot_error)], 'k--', label=str(val) + " Data Pts")
        ax3.legend()
        plt.savefig(f"{fp}/individual_overlapping.png")
    
    return t_vals, tot_error



if __name__=="__main__":
    #single_dim_test()
    #multi_dim_test()
    #data_sensitivity_test(
    #    plot=True, 
    #    fp="/home/hunter/Pictures/dmp_counts",
    #    max_t=101
    #)
    """
    fig2, ax2 = plt.subplots()
    ax2.set_title("Total Error as function of RBFs")
    ax2.set(xlabel = "# Data Points")
    ax2.set(ylabel = "Sum Squared Error")
    plot_set = [50]
    tot_error_max = -1
    for nRBFs in [5,10,20,30,50,100]:
        t_vals, tot_error = data_sensitivity_test(
            plot=False, 
            fp="/home/hunter/Pictures/dmp_counts",
            max_t=21,
            nRBFs=nRBFs
        )
        ax2.plot(t_vals, tot_error, label=f"{nRBFs} RBFs")
        tot_error_max = max(tot_error_max, max(tot_error))
    for val in plot_set:
        ax2.plot([val,val],[0,tot_error_max], 'k--', label=str(val) + " Data Pts")
    ax2.legend()
    plt.savefig(f"/home/hunter/Pictures/dmp_counts/total_error_comp.png")"
    """
    #assert 1 == 0
    t_vals = [k * 10 for k in range(2, 101)]
    for t_val in [20, 500]:#[20, 30, 50, 200, 500, t_vals[-1]]:
        test(num_dims=3,num_envs=4, tmax=0.1, dt=0.5/t_val,fp="/home/hunter/Pictures/dmp_counts")
        assert 1 == 0
    for i in [1,4]:
        for j in [1,3]:
            test(num_dims=j,num_envs=i)