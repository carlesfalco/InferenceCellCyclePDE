import numpy as np
from scipy.integrate import ode

# Solve model in polar coordinates, given initial conditions and model parameters

class cell_cycle_model_radial2d_linear_diffusion:

    def __init__(self,rmax,N,dt,init_rho_1,init_rho_2,tmax,D,kr,kg,K1,K2):
        
        self.rmax = rmax
        self.N = N
        self.dt = dt
        self.dr = rmax/N
        self.D = D
        self.K1 = K1
        self.K2 = K2
        self.kr = kr
        self.kg = kg
        self.tmax = tmax
        
        self.rhalf = np.array([self.dr*(k+1/2) for k in range(self.N + 1)])
        self.rspan = np.array([self.dr*(k) for k in range(self.N + 2)])
        self.A = np.array([np.pi*self.dr**2*(2*i + 1) for i in range(self.N + 1)])
        
        self.rho = [np.concatenate([init_rho_1,init_rho_2])]
        
    def f(self,p):
        return (1 - (p/self.K1))*(p < self.K1)
        #return 1*(p < self.K1) for Heaviside model
        
    def g(self,p):
        return (1 - (p/self.K2))*(p < self.K2)
        #return 1*(p < self.K2) for Heaviside model
    
    def drho(self,p):
        drho = (p[1:] - p[:-1])/self.dr
        return np.concatenate([[0],drho,[0]]) # dummy drho[0]
    
    def dxdt_diffusion(self,p):
        dp = self.drho(p)
        diff = np.zeros(p.shape)
        diff[0] = 2*np.pi/self.A[0]*self.rspan[1]*dp[1]
        diff[1:] = 2*np.pi/self.A[1:]*(self.rspan[2:]*dp[2:] - self.rspan[1:-1]*dp[1:-1])
        return diff
    
    def dxdt_reaction(self,p1,p2):
        return np.concatenate([ -self.kr*p1*self.f(p1 + p2) + 2*self.kg*p2*self.g(p1 + p2), 
                               self.kr*p1*self.f(p1 + p2) - self.kg*p2*self.g(p1 + p2) ])
    def dxdt(self,t,p):
        p1,p2 = np.split(p,2)
        diff1 = self.dxdt_diffusion(p1)*self.D
        diff2 = self.dxdt_diffusion(p2)*self.D
        return np.concatenate([diff1,diff2]) + self.dxdt_reaction(p1,p2)
    
    def solve(self):
        t = 0
        solODE = ode(self.dxdt).set_integrator('vode')
        solODE.set_initial_value(self.rho[0],0)
        while t < self.tmax:
            #print('t = %.2f'% t,end = '\r') # for time
            t += self.dt
            r1, r2 = np.split(solODE.integrate(t),2)
            self.rho.append(np.concatenate([r1,r2]))
        return np.array(self.rho)[:,:self.N + 1], np.array(self.rho)[:,self.N + 1:]
    
    
    
    
    
class cell_cycle_model1d_linear_diffusion: # Solver for the 1D model, used for TWs and scratch assay data

    def __init__(self,L,N,dt,init_rho_1,init_rho_2,tmax,Dr,Dg,kr,kg,K,K2):
        
        self.L = L
        self.N = N
        self.dt = dt
        self.dx = L/N
        self.Dr = Dr
        self.Dg = Dg
        self.K1 = K1
        self.K2 = K2
        self.kr = kr
        self.kg = kg
        self.tmax = tmax
        
        self.rho = [np.array([init_rho_1,init_rho_2])]
        
    def f(self,p):
        return (1 - (p/self.K1))*(p < self.K1)
        
    def g(self,p):
        return (1 - (p/self.K2))*(p < self.K2)
    
    def dxdt(self, t,y):
        Dr, Dg, kr, kg = self.Dr, self.Dg, self.kr, self.kg
        dx = self.dx
        r, g = y[:N], y[N:]
        drdt, dgdt = np.zeros(r.shape), np.zeros(g.shape)
        drdt[0] = 2*Dr*(r[1]-r[0])/dx**2 - kr*r[0]*self.f(r[0] + g[0]) + 2*kg*g[0]*self.g(r[0] + g[0])
        dgdt[0] = 2*Dg*(g[1]-g[0])/dx**2 + kr*r[0]*self.f(r[0] + g[0]) - kg*g[0]*self.g(r[0] + g[0])
        drdt[-1] = 2*Dr*(r[-2]-r[-1])/dx**2 - kr*r[-1]*self.f(r[-1] + g[-1]) + 2*kg*g[-1]*self.g(r[-1] + g[-1])
        dgdt[-1] = 2*Dg*(g[-2]-g[-1])/dx**2 + kr*r[-1]+0*self.f(r[-1] + g[-1]) - kg*g[-1]*self.g(r[-1] + g[-1])
        drdt[1:-1] = Dr*(r[2:] - 2*r[1:-1] + r[:-2])/dx**2 - kr*r[1:-1]+self.f(r[1:-1] + g[1:-1]) + 2*kg*g[1:-1]*self.g(r[1:-1] + g[1:-1])
        dgdt[1:-1] = Dg*(g[2:] - 2*g[1:-1] + g[:-2])/dx**2 + kr*r[1:-1]+self.f(r[1:-1] + g[1:-1]) - kg*g[1:-1]*self.g(r[1:-1] + g[1:-1])
        return np.concatenate([drdt,dgdt])

    
    def solve(self):
        solODE = ode(self.dxdt).set_integrator('vode')
        solODE.set_initial_value(np.concatenate(self.rho[0]),0)
        t = 0
        while t < self.tmax:
            #print('t = %.2f'% t,end = '\r')
            t += self.dt
            r1, r2 = np.split(solODE.integrate(t),2)
            self.rho.append(np.array([r1,r2]))
        return np.array(self.rho)[:,0], np.array(self.rho)[:,1]
        