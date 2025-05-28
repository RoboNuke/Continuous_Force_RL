from math import exp
import matplotlib.pyplot as plt
import numpy as np
from wrappers.DMP.rbf import RBF
from wrappers.DMP.cs import CS


class DMP_1D():
    def __init__(self, betaY=1, nRBF=100, alphaX=-1.0, dt = 0.001, data=None):
        self.goal = None
        self.start = None
        if data == None:
            self.nRBF = nRBF
            self.ay = 4 * betaY
            self.by = betaY
            self.ax = alphaX
            self.dt = dt
            self.w = [1.0 for x in range(self.nRBF)]
            self.cs = CS(self.ax, self.dt)
            self.centerRBFs()
        else:
            self.by = data['by']
            self.ay = 4 * self.by
            self.nRBF = len(data['weights'])
            self.w = data['weights']
            self.RBFs = []
            for i in range(self.nRBF):
                c = data['centers'][i]
                h = data['widths'][i]
                self.RBFs.append(RBF(c,h))

            self.dt = data['dt']
            self.ax = data['ax']
            self.cs = CS(self.ax, self.dt)
         
    def centerRBFs(self):
        self.RBFs = []
        #des_c = np.linspace(min(self.cs.xPath), max(self.cs.xPath), self.nRBF)
        des_c = np.linspace(0, self.cs.run_time, self.nRBF)
        #print(len(self.cs.xPath))
        for i in range(self.nRBF):
            c = np.exp(-self.cs.ax * des_c[i])
            #c = des_c[i]
            #h = (self.nRBF) / (c ** 2)
            h = self.nRBF ** 1.5 / c / self.cs.ax
            self.RBFs.append(RBF(c,h))
            #print(des_c[i], c)
            #print(c, h)


    def plotActivations(self, with_x = True, with_w = True):
        self.actives = [[] for y in range(self.nRBF)]

        x_track = self.cs.xPath

        ts = np.linspace(0, self.cs.run_time, self.cs.steps)
        for t in ts:
            x = exp(-self.ax * t)
            for i in range(self.nRBF):
                if(with_w):
                    self.actives[i].append(self.RBFs[i].eval(x) * self.w[i])
                else:
                    self.actives[i].append(self.RBFs[i].eval(x))

        for i in range(self.nRBF):
            #print(f"c:{self.RBFs[i].c}, h:{self.RBFs[i].h}")
            plt.plot( ts, self.actives[i])
        if(with_x):
            plt.plot(ts, x_track)
        #plt.xlim(max(x_track), min(x_track))

    def learnWeights(self, t, y, dy, ddy):
        g = y[-1]
        # evenly space path along the time
        self.timesteps = int(t[-1] / self.dt)
        path = np.zeros(self.timesteps)

        import scipy.interpolate
        path_gen = scipy.interpolate.interp1d(t, y)
        for t in range(self.timesteps):
            path[t] = path_gen(t * self.dt)

        y = path
        #plt.plot(y, '-r')
        #plt.plot(ay)
        #plt.show()
        
        # estimate the gradients
        ydot = np.gradient(y)/self.dt
        ydotdot = np.gradient(ydot)/self.dt
        #ydot= dy
        #ydotdot = ddy
        #print("ydot:", min(ydot), max(ydot))
        #print("ydotdot:", min(ydotdot), max(ydotdot))
        """
        plt.plot(np.linspace(0,len(dy), len(ydot)),ydot, '-r')
        plt.plot(dy)
        plt.show()
        plt.plot(np.linspace(0,len(dy), len(ydot)), ydotdot, '-r')
        plt.plot(ddy)
        plt.show()
        """
        #y = ay
        #ydot = dy#/self.dt
        #ydotdot = ddy#/(self.dt**2)

        x = np.array(self.cs.rollout(self.timesteps))
        rbMats = [self.RBFs[i].theeMat(self.cs.xPath) for i in range(self.nRBF)]

        fd = ydotdot - self.ay * (self.by * (g - y) - ydot)
        #fd = fd.T
        #x = np.array(self.cs.xPath)
        #print(fd.shape, rbMats[0].shape, y.shape, x.shape)
        
        for i in range(len(self.w)):
            bot = np.sum( x ** 2 * rbMats[i])
            top = np.sum( x * rbMats[i] * fd)
            self.w[i] = top / bot
        #print(g - y[0])
        if abs(g - y[0]) > 0.0001:
            #print("in g thing")
            for i in range(self.nRBF):
                self.w[i]/= (g-y[0])


    def calcF(self, x):
        top = 0
        bot = 0
        for i in range(len(self.w)):
            thee = self.RBFs[i].eval(x)
            top += thee * self.w[i] 
            bot += thee
        return top/bot

    def rollOut(self, y0, g, tau=1, dy0=10, ddy0=-9):
        y = y0
        ydot = dy0
        ydotdot = ddy0
        t = 0.0
        x = 1.0
        z = [y0]
        dz = [ydot]
        ddz = [ydotdot]
        #print(self.dt)
        ts = [0.0]
        #print(f"Total Time:{self.cs.run_time * tau}")
        while( self.cs.run_time * tau  - t > 0.0001):
            ydotdot = self.ay * (self.by * (g -  y) - ydot) +  x*(g-y0)*self.calcF(x)
            #print(self.dt / tau, self.dt)
            ydot += ydotdot * (self.dt / tau)
            y += ydot * (self.dt / tau)
            z.append(y)
            dz.append(ydot)
            ddz.append(ydotdot)
            xdot = -self.ax * x / tau
            t+=self.dt
            ts.append(t)
            #x = exp(-self.ax * t)
            x += xdot * self.dt

            #print(x,t, self.dt)
        return(ts, z, dz, ddz)

def plotSub(ax, t, org, cpy, tit="DMP", ylab="Function"):
    ax.plot(t, org)
    ax.plot(np.linspace(t[0], t[-1], len(cpy)), cpy,'r--')
    ax.set(ylabel=ylab)
    ax.set_title(tit,  fontsize=20)


def main():
    dt = 0.01
    tmax = 5.0
    dmp = DMP_1D(betaY=25.0/4.0, nRBF=1000, alphaX=1.0, dt = dt)
    n = int(tmax / dt)
    print(n)
    t = np.arange(0, tmax, dt)
    #y = t ** 2
    #dy = 2 * t
    #ddy = 2 * np.ones(n)
    #g = t[-1] ** 2
    
    #y = (t - 2.5) ** 2 + 2.5 * t 
    y = np.sin(10*t) + np.cos(3*t)
    dy = 10*np.cos(10*t) - 3*np.sin(3*t)
    ddy = -100 * np.sin(10*t) - 9 * np.cos(3*t)
    #y = np.zeros(t.shape)
    #y[int(len(t)/2):] = 1
    #y = np.sin(t)
    g = y[-1]
    print("Learning Weights")
    dmp.learnWeights(t, y, dy, ddy)
    #ow = np.array( [-3.51377887e+02,-3.68127514e+02,-3.74511242e+02,-3.48613849e+02,-2.80977915e+02,-1.88114903e+02,-9.86042973e+01,-3.45547045e+01,-1.81576319e+00,6.51977254e+00] )
    
    #print("Weights:", np.array(dmp.w))
    #print(ow)
    #print(dmp.w - ow)
    #print("Starting Rollout")
    #print(y[0], g)
    print("Rolling out trajectory")
    tau = 1
    scale = 1
    #ts, z = dmp.rollOut(scale * y[0], scale * g, tau)
    ts, z, dz, ddz = dmp.rollOut(scale * y[0], scale * g, tau)
    #dmp.plotActivations()
    #print(z)
    single = False
    if single:
        plt.plot(ts, z )
        #plt.plot(ts, k)
        plt.plot(np.linspace(0, tau, len(y)), y* scale,'r--')
        plt.xlabel("time(s)")
        plt.ylabel("Function")
        plt.title('DMP of sin(10t) + cos(3t)')
        plt.legend(['Learned Trajectory', 'Original Function'])
        plt.tight_layout()
        plt.show()
    else:
        #print(min(y), max(y))
        #print(min(z), max(z))
        #print(min(dy), max(dy))
        #print(min(dz), max(dz))
        #print(min(ddy), max(ddy))
        #print(min(ddz), max(ddz))
        #print(y[0], y[-1])
        fig, axs = plt.subplots(3)
        fig.set_figwidth(800/96)
        fig.set_figheight(1000/96)
        fig.tight_layout(pad=5.0)
        plotSub(axs[0], t, y, z,"Position DMP", "Position")
        plotSub(axs[1], t, dy, dz, "Velocity DMP", "Velocity")
        plotSub(axs[2], t, ddy, ddz, "Accel DMP", "Acceleration")
        plt.xlabel("time (s)")

        plt.legend(['Original Function', 'Learned Trajectory'])
        plt.show()

       
    
if __name__ == '__main__':
    main()

"""
    Error Term
    display multiple graphs with comparison
    Abstract class?
"""