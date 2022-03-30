import numpy as np
from anesthetic import MCMCSamples, NestedSamples
import tqdm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import sys
from scipy.stats import powerlaw
from scipy.special import logsumexp


class NS(object):
        
    def logX_powerlaw(self,logL,nlive):
        ndead = len(logL)
        t = powerlaw(nlive).rvs(ndead)
        logX = np.log(t).cumsum()
        return logX
    
    
    def logZ(self,logL,logX):
        logsum_L=logsumexp([logL[1:],logL[:-1]],axis=0)
        logdiff_X=logsumexp([logX[1:],logX[:-1]],axis=0,b=np.array([-1,1])[:,None])
        logQ=logsum_L+logdiff_X-np.log(2)
        logZ=logsumexp(logQ)
        return logZ
    
    def logp(self,logL,logX_):
        logX = np.concatenate((0,logX_,-np.inf),axis=None)
        logdiff_X=logsumexp([logX[2:],logX[:-2]],axis=0,b=np.array([-1,1])[:,None])
        logp=logL+logdiff_X-np.log(2)
        return logp-logsumexp(logp)
    
    
    def NS2d(self):
        MHns = NestedSamples(data=self.data, columns=self.columns, logL=self.logL, logL_birth=self.logL_birth, tex=self.tex)
        return MHns.plot_2d(['x0','x1'],alpha=0.5)
    
    def MCMC2d(self,parameters,label_):
        return self.MCMC.plot_2d(parameters,alpha = 0.5,label=label_)
   
    def MCMCstd(self):
        return self.MCMC['x0'].std()



class OrthodoxNS(NS):
    def __init__(self,loglikelihood,sigma,prior_bounds, ndims=2, nlive=125, num_repeats=10):
        """Metropolis Hastings Nested Sampling run"""
        self.ndims = ndims
        self.columns = ['x%i' % i for i in range(ndims)]
        self.tex = {p: '$x_%i$' % i  for i, p in enumerate(self.columns)}
        self.nlive=nlive
        low= prior_bounds[0]
        high=prior_bounds[1]
        live_points = np.random.uniform(low=low, high=high, size=(nlive, ndims))
        live_likes = np.array([loglikelihood(x,) for x in live_points])
        live_birth_likes = np.ones(nlive) * -np.inf
    
        dead_points = []
        dead_likes = []
        birth_likes = []
        for _ in tqdm.tqdm(range(nlive*11)):
            i = np.argmin(live_likes)
            Lmin = live_likes[i]
            dead_points.append(live_points[i].copy())
            dead_likes.append(live_likes[i])
            birth_likes.append(live_birth_likes[i])
            live_birth_likes[i] = Lmin
            while live_likes[i] <= Lmin:
                live_points[i, :] = np.random.uniform(low=low, high=high, size=ndims) 
                live_likes[i] = loglikelihood(live_points[i])
        self.data, self.logL, self.logL_birth, self.live, self.live_logL, self.live_logL_birth =  np.array(dead_points), np.array(dead_likes), np.array(birth_likes), live_points, live_likes, live_birth_likes
        self.weights=np.exp(self.logp(self.logL,self.logX_powerlaw(self.logL,self.nlive)))
        self.MCMC = MCMCSamples(data=self.data, columns=self.columns, logL=self.logL, weights=self.weights, tex=self.tex)
        


class MetropolisNS(NS):
    def __init__(self,loglikelihood,sigma,prior_bounds, ndims=2, nlive=125, num_repeats=10):
        """Metropolis Hastings Nested Sampling run"""
        self.ndims = ndims
        self.columns = ['x%i' % i for i in range(ndims)]
        self.tex = {p: '$x_%i$' % i  for i, p in enumerate(self.columns)}
        self.nlive=nlive
        low= prior_bounds[0]
        high=prior_bounds[1]
        live_points = np.random.uniform(low=low, high=high, size=(nlive, ndims))
        live_likes = np.array([loglikelihood(x,) for x in live_points])
        live_birth_likes = np.ones(nlive) * -np.inf
    
        dead_points = []
        dead_likes = []
        birth_likes = []
        for _ in tqdm.tqdm(range(nlive*11)):
            i = np.argmin(live_likes)
            Lmin = live_likes[i]
            dead_points.append(live_points[i].copy())
            dead_likes.append(live_likes[i])
            birth_likes.append(live_birth_likes[i])
            live_birth_likes[i] = Lmin
            for gg in range(num_repeats):
                while True:
                    live_point = live_points[i]+np.random.multivariate_normal(mean=np.zeros(len(live_points[0])), cov=np.cov(live_points.T)/2)
                    if loglikelihood(live_point) > Lmin and np.all([live_point>low,live_point<high]):
                        break
                live_points[i, :] = live_point
                live_likes[i] = loglikelihood(live_points[i])
        # for _ in tqdm.tqdm(range(nlive*11)):
        #     i = np.argmin(live_likes)
        #     Lmin = live_likes[i]
        #     dead_points.append(live_points[i].copy())
        #     dead_likes.append(live_likes[i])
        #     birth_likes.append(live_birth_likes[i])
        #     live_birth_likes[i] = Lmin
        #     while live_likes[i] <= Lmin:
        #         live_points[i, :] = np.random.uniform(low=low, high=high, size=ndims) 
        #         live_likes[i] = loglikelihood(live_points[i])
        self.data, self.logL, self.logL_birth, self.live, self.live_logL, self.live_logL_birth =  np.array(dead_points), np.array(dead_likes), np.array(birth_likes), live_points, live_likes, live_birth_likes
        self.weights=np.exp(self.logp(self.logL,self.logX_powerlaw(self.logL,self.nlive)))
        self.MCMC = MCMCSamples(data=self.data, columns=self.columns, logL=self.logL, weights=self.weights, tex=self.tex)
    
    
    # def weights(self,nlive=125,ndead=1376):
    #    t = powerlaw(nlive).rvs(ndead)
    #    logX = np.log(t).cumsum()
    #    X = np.concatenate((1,np.exp(logX)),axis=None)
    #    w = 0.5*(X[0:-2]-X[2:])
    #    return w
        
    

