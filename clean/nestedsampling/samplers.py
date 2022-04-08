import numpy as np
from anesthetic import MCMCSamples, NestedSamples
import tqdm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import sys
from scipy.stats import powerlaw
from scipy.special import logsumexp


class NSrun():
    def __init__(self,loglikelihood,prior_bounds, ndims=2, nlive=125,Metropolis=False,num_repeats=125, tol=1e-30):
        """orthodox Nested Sampling run"""
        self.ndims = ndims
        self.columns = ['x%i' % i for i in range(ndims)]
        self.tex = {p: '$x_%i$' % i  for i, p in enumerate(self.columns)}
        self.nlive=nlive
        self.loglikelihood=loglikelihood
        self.tolerance_breaks=0
        self.tol= tol
        self.logZ_tester=0
        print('tolerance chosen as'+str(self.tol))
        low= prior_bounds[0]
        high=prior_bounds[1]
        live_points = np.random.uniform(low=low, high=high, size=(nlive, ndims))
        live_likes = np.array([self.loglikelihood(x,) for x in live_points])
        live_birth_likes = np.ones(nlive) * -np.inf
        dead_points = []
        dead_likes = []
        birth_likes = []
        if Metropolis:
            for _ in tqdm.tqdm(range(nlive*11)):
                i = np.argmin(live_likes)
                Lmin = live_likes[i]
                dead_points.append(live_points[i].copy())
                dead_likes.append(live_likes[i])
                birth_likes.append(live_birth_likes[i])
                live_birth_likes[i] = Lmin
                j= np.random.randint(low=0, high= len(live_points))
                wandering_point= live_points[j]
                for gg in range(num_repeats):
                    while True:
                        live_point = wandering_point+np.random.multivariate_normal(mean=np.zeros(len(live_points[0])), cov=np.cov(live_points.T)/42)
                        if self.loglikelihood(live_point) > Lmin and np.all([live_point>low,live_point<high]):
                            break
                    wandering_point = live_point
                """Below we check if breaking criterion is met"""
                if _>=1:
                    self.logL = np.array(dead_likes)
                    self.logX_powerlaw()
                    self.frac= np.log(0.5*(np.exp(self.logL[-1])+np.exp(self.logL[-2]))*(np.exp(self.logX[-2])-np.exp(self.logX[-1])))
                    self.logZ_tester += self.frac
                    cond = self.breaking_criterion()
                    if cond:
                        print(cond)
                        break
                live_points[i, :] = live_point
                live_likes[i] = self.loglikelihood(live_points[i])
            print('random walk steps are'+str(num_repeats))
        else:
            for _ in tqdm.tqdm(range(nlive*11)):
                i = np.argmin(live_likes)
                Lmin = live_likes[i]
                dead_points.append(live_points[i].copy())
                dead_likes.append(live_likes[i])
                birth_likes.append(live_birth_likes[i])
                live_birth_likes[i] = Lmin
                while live_likes[i] <= Lmin:
                    live_points[i, :] = np.random.uniform(low=low, high=high, size=ndims) 
                    live_likes[i] = self.loglikelihood(live_points[i])
                """Below we check if breaking criterion is met"""
                if _>=1:
                    self.logL = np.array(dead_likes)
                    self.logX_powerlaw()
                    self.frac= np.log(0.5*(np.exp(self.logL[-1])+np.exp(self.logL[-2]))*(np.exp(self.logX[-2])-np.exp(self.logX[-1])))
                    self.logZ_tester += self.frac
                    cond = self.breaking_criterion()
                    if cond:
                        print(cond)
                        break
        self.data, self.logL, self.logL_birth, self.live, self.live_logL, self.live_logL_birth =  np.array(dead_points), np.array(dead_likes), np.array(birth_likes), live_points, live_likes, live_birth_likes
        print(str(self.logZ_tester)+'should be same as'+str(self.logZval))
              
    def breaking_criterion(self):
        self.logX_powerlaw()
        #print('wills logZ is'+str(logZ__))
        self.frac= np.log(0.5*(np.exp(self.logL[-1])+np.exp(self.logL[-2]))*(np.exp(self.logX[-2])-np.exp(self.logX[-1])))
        self.logZval = self.logZ()
        self.logZ_increment= (self.frac)/(self.logZval)
        #print('logZ_increment is'+str(self.logZ_increment))
        #print('logZ is'+str(self.logZval))
        if abs(self.logZ_increment)<self.tol:
            self.tolerance_breaks+=1
           # print('consecutive tolerance braeks reached'+str(self.tolerance_breaks))
            if self.tolerance_breaks>=5:
                print('we hit 5 consecutive tolerance breaks')
                return True
            else:
                return False
        else:
            self.tolerance_breaks=0
            return False
            
            
    def logX_powerlaw(self):
        self.ndead = len(self.logL)
        t = powerlaw(self.nlive).rvs(self.ndead)
        self.logX = np.log(t).cumsum()
        return self.logX
    
    
    def logZ_int(self,reps):
        self.MHns = NestedSamples(data=self.data, columns=self.columns, logL=self.logL, logL_birth=self.logL_birth, tex=self.tex)
        return self.MHns.logZ(reps)
        
    
    def logZ(self):
        self.logX_powerlaw()
        logsum_L=logsumexp([self.logL[1:],self.logL[:-1]],axis=0)
        logdiff_X=logsumexp([self.logX[1:],self.logX[:-1]],axis=0,b=np.array([-1,1])[:,None])
        logQ=logsum_L+logdiff_X-np.log(2)
        logZ=logsumexp(logQ)
        return logZ
    
    def logp(self,logL,logX_):
        logX = np.concatenate((0,logX_,-np.inf),axis=None)
        logdiff_X=logsumexp([logX[2:],logX[:-2]],axis=0,b=np.array([-1,1])[:,None])
        logp=logL+logdiff_X-np.log(2)
        return logp-logsumexp(logp)
    
    def NS2d(self,parameters,label_):
        self.MHns = NestedSamples(data=self.data, columns=self.columns, logL=self.logL, logL_birth=self.logL_birth, tex=self.tex)
        return self.MHns.plot_2d(parameters,alpha=0.5,label=label_)
    
    def MCMC2d(self,parameters,label_):
        self.weights=np.exp(self.logp(self.logL,self.logX_powerlaw()))
        self.MCMC = MCMCSamples(data=self.data, columns=self.columns, logL=self.logL, weights=self.weights, tex=self.tex)
        return self.MCMC.plot_2d(parameters,alpha = 0.5,label=label_)
    
    def MCMCstd(self):
        self.weights=np.exp(self.logp(self.logL,self.logX_powerlaw()))
        self.MCMC = MCMCSamples(data=self.data, columns=self.columns, logL=self.logL, weights=self.weights, tex=self.tex)
        return self.MCMC['x0'].std()

