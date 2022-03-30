#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 17:48:32 2022

@author: sahibzadaallahyar
"""

import numpy as np 
from anesthetic import MCMCSamples, NestedSamples
import tqdm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import sys
from scipy.stats import powerlaw
from scipy.special import logsumexp

ndims = 2
columns = ['x%i' % i for i in range(ndims)]
tex = {p: '$x_%i$' % i  for i, p in enumerate(columns)}

np.random.seed(0)

nlive=125
m_true = 1
c_true = 0.5
sigma = 0.1
N = 100

# def f(data_w_noise, theta):
#      m, c = theta
#      x = data_w_noise[:,0]
#      noise = data_w_noise[:,1]
#      return m * x + c + noise

# def loglikelihood(theta):
#       m, c = theta
#       y = m * data_x + c
#       logL = -(np.log(2*np.pi*sigma**2)/2 + (data_y - y)**2/2/sigma**2).sum()
#       return logL


# def loglikelihood(theta):
#     number_of_rows= data_noise.shape[0]
#     random_indices= np.random.choice(number_of_rows,size=N,replace=False)
#     random_rows = data_noise[random_indices, :]
#     m, c = theta
#     data_y_sub = f(random_rows, [m, c])
#     y1 = m * random_rows[:,0] + c
#     logL = -(np.log(2*np.pi*sigma**2)/2 + (data_y_sub - y1)**2/2/sigma**2).sum()
#     return logL



# data_x = np.random.uniform(-1, 1, N)
# noise = np.random.randn(N)*sigma
# data_noise= np.vstack((data_x,noise)).T
# data_y = f(data_noise, [m_true, c_true])

def f(x, theta):
      m, c = theta
      return m * x + c

data_x = np.random.uniform(-1, 1, N)
data_y = f(data_x, [m_true, c_true]) + np.random.randn(N)*sigma

def loglikelihood(theta,Nsub=10):
      y = f(data_x, theta)
      logL = -(np.log(2*np.pi*sigma**2)/2 + (data_y - y)**2/2/sigma**2).sum()
      return logL

def nondet_loglikelihood(theta,Nsub = 10):
     i = np.random.choice(len(data_x),Nsub,replace=False)
     y = f(data_x[i],theta)
     logL = -(np.log(2*np.pi*sigma**2)/2 + (data_y[i] - y)**2/2/sigma**2).sum()
     return logL



def ns_sim_mh(logL, sub, ndims=2, nlive=125, num_repeats=10):
    """Metropolis Hastings Nested Sampling run"""
    low=[0,0]
    high=[2,1]
    live_points = np.random.uniform(low=low, high=high, size=(nlive, ndims))
    live_likes = np.array([logL(x,Nsub=sub) for x in live_points])
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
                if logL(live_point,Nsub=sub) > Lmin:
                    break
            live_points[i, :] = live_point
            live_likes[i] = logL(live_points[i],Nsub=sub)
    return np.array(dead_points), np.array(dead_likes), np.array(birth_likes), live_points, live_likes, live_birth_likes



def ns_sim(logL,ndims=2, nlive=125):
    """Brute force Nested Sampling run"""
    low=(0,0)
    high=(2,1)
    live_points = np.random.uniform(low=low, high=high, size=(nlive, ndims))
    live_likes = np.array([logL(x) for x in live_points])
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
            live_likes[i] = logL(live_points[i])
    return np.array(dead_points), np.array(dead_likes), np.array(birth_likes), live_points, live_likes, live_birth_likes




def weights(nlive=125,ndead=1376):
   t = powerlaw(nlive).rvs(ndead)
   logX = np.log(t).cumsum()
   X = np.concatenate((1,np.exp(logX)),axis=None)
   w = 0.5*(X[0:-2]-X[2:])
   return w
    
def logX_powerlaw(logL,nlive):
    ndead = len(logL)
    t = powerlaw(nlive).rvs(ndead)
    logX = np.log(t).cumsum()
    return logX


def logZ(logL,logX):
    logsum_L=logsumexp([logL[1:],logL[:-1]],axis=0)
    logdiff_X=logsumexp([logX[1:],logX[:-1]],axis=0,b=np.array([-1,1])[:,None])
    logQ=logsum_L+logdiff_X-np.log(2)
    logZ=logsumexp(logQ)
    return logZ

def logp(logL,logX_):
    logX = np.concatenate((0,logX_,-np.inf),axis=None)
    logdiff_X=logsumexp([logX[2:],logX[:-2]],axis=0,b=np.array([-1,1])[:,None])
    logp=logL+logdiff_X-np.log(2)
    return logp-logsumexp(logp)


data, logL, logL_birth, live, live_logL, live_logL_birth = ns_sim(ndims=ndims,logL=nondet_loglikelihood)

MHns2 = NestedSamples(data=data, columns=columns, logL=logL, logL_birth=logL_birth, tex=tex)
live_MHns2 = NestedSamples(data=live, columns=columns, logL=live_logL, logL_birth=live_logL_birth, tex=tex)
fig, ax = MHns2.plot_2d(['x0','x1'],label='Non-deterministic N=10')


################################

data, logL, logL_birth, live, live_logL, live_logL_birth = ns_sim(ndims=ndims,logL=loglikelihood)

MHns3 = NestedSamples(data=data, columns=columns, logL=logL, logL_birth=logL_birth, tex=tex)
live_MHns2 = NestedSamples(data=live, columns=columns, logL=live_logL, logL_birth=live_logL_birth, tex=tex)

MHns3.plot_2d(axes= ax,label='Deterministic N=100',alpha=0.5)

######################################



N=10
data_x = np.random.uniform(-1, 1, N)
data_y = f(data_x, [m_true, c_true]) + np.random.randn(N)*sigma


data, logL, logL_birth, live, live_logL, live_logL_birth = ns_sim(ndims=ndims,logL=loglikelihood)

MHns4 = NestedSamples(data=data, columns=columns, logL=logL, logL_birth=logL_birth, tex=tex)
live_MHns2 = NestedSamples(data=live, columns=columns, logL=live_logL, logL_birth=live_logL_birth, tex=tex)
MHns4.plot_2d(axes= ax,alpha = 0.5,label='Deterministic N=10')



##################################################

weights = weights()

data, logL, logL_birth, live, live_logL, live_logL_birth = ns_sim(ndims=ndims,logL=nondet_loglikelihood)



weights=np.exp(logp(logL,logX_powerlaw(logL,nlive)))

MCMC = MCMCSamples(data=data, columns=columns, logL=logL, weights=weights, tex=tex)
live_MHns2 = NestedSamples(data=live, columns=columns, logL=live_logL, logL_birth=live_logL_birth, tex=tex)

MCMC.plot_2d(axes= ax,alpha = 0.5,label='Non-Det N=10 Weighted MCMC')


handles, labels = ax['x0']['x1'].get_legend_handles_labels()
leg = fig.legend(handles, labels)
fig.tight_layout()

#################################


plt.figure(0)

plt.hist(MHns3.logZ(100),label= 'Deterministic N=100')
plt.hist(MHns2.logZ(100),label= 'Non-Deterministic N=10')
plt.hist(MHns4.logZ(100),label= 'Deterministic N=10')
plt.legend(loc="upper left")
sys.exit(0)

















data, logL, logL_birth, live, live_logL, live_logL_birth = ns_sim(ndims=ndims, logL=nondet_loglikelihood())


ns2 = NestedSamples(data=data, columns=columns, logL=logL, logL_birth=logL_birth, tex=tex)
live_ns2 = NestedSamples(data=live, columns=columns, logL=live_logL, logL_birth=live_logL_birth, tex=tex)


ns2.plot_2d(['x0','x1'])


sys.exit(0)


from pypolychord import run_polychord, PolyChordSettings
from pypolychord.priors import UniformPrior

def prior(cube):
     m = UniformPrior(-5,5)(cube[0])
     c = UniformPrior(-5,5)(cube[1])
     return [m, c]

nDims = 2
nDerived = 0
settings = PolyChordSettings(nDims, nDerived)
settings.read_resume = False
run_polychord(loglikelihood, nDims, nDerived, settings, prior)

from anesthetic import NestedSamples
import os
root = os.path.join(settings.base_dir, settings.file_root)
samples = NestedSamples(root=root, columns=['m','c'])
samples.plot_2d(['m','c'])
plt.savefig('anesthetic.pdf')

from fgivenx import plot_contours
x = np.linspace(-1,1,101)
fig, ax = plt.subplots()
plt.errorbar(data_x, data_y, sigma, fmt='.')
plot_contours(f, x, samples[['m','c']], weights=samples.weights)
plt.savefig('fgivenx.pdf')