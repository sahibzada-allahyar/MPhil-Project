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


ndims = 2
columns = ['x%i' % i for i in range(ndims)]
tex = {p: '$x_%i$' % i  for i, p in enumerate(columns)}


np.random.seed(0)

m_true = 1
c_true = 0.5
sigma = 0.1
N = 100

def f(x, theta):
     m, c = theta
     return m * x + c

data_x = np.random.uniform(-1, 1, N)
noise = np.random.randn(N)*sigma
data_y = f(data_x, [m_true, c_true]) + noise

def loglikelihood(theta):
     m, c = theta
     y = m * data_x + c
     logL = -(np.log(2*np.pi*sigma**2)/2 + (data_y - y)**2/2/sigma**2).sum()
     return logL


def nondet_loglikelihood(theta):
    data_y = f(np.random.choice(data_x,replace=False), [m_true, c_true]) + noise 
    m, c = theta
    y = m * data_x + c
    logL = -(np.log(2*np.pi*sigma**2)/2 + (data_y - y)**2/2/sigma**2).sum()
    return logL


def ns_sim_mh(ndims=2, nlive=125,num_repeats=10):
    """Metropolis Hastings Nested Sampling run"""
    low=(0,0)
    high=(2,1)
    live_points = np.random.uniform(low=low, high=high, size=(nlive, ndims))
    live_likes = np.array([loglikelihood(x) for x in live_points])
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
            live_point = live_points[i]+np.random.multivariate_normal(mean=np.zeros(len(live_points[0])), cov=np.cov(live_points.T)/2)
            while loglikelihood(live_point) <= Lmin:
                live_point = live_points[i]+np.random.multivariate_normal(mean=np.zeros(len(live_points[0])), cov=np.cov(live_points.T)/2)
            live_points[i, :] = live_point
            live_likes[i] = loglikelihood(live_points[i])
    return np.array(dead_points), np.array(dead_likes), np.array(birth_likes), live_points, live_likes, live_birth_likes



def ns_sim(ndims=2, nlive=125):
    """Brute force Nested Sampling run"""
    low=(0,0)
    high=(2,1)
    live_points = np.random.uniform(low=low, high=high, size=(nlive, ndims))
    live_likes = np.array([loglikelihood(x) for x in live_points])
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
    return np.array(dead_points), np.array(dead_likes), np.array(birth_likes), live_points, live_likes, live_birth_likes



data, logL, logL_birth, live, live_logL, live_logL_birth = ns_sim_mh()

MHns2 = NestedSamples(data=data, columns=columns, logL=logL, logL_birth=logL_birth, tex=tex)
live_MHns2 = NestedSamples(data=live, columns=columns, logL=live_logL, logL_birth=live_logL_birth, tex=tex)
MHns2.plot_2d(['x0','x1'])

data, logL, logL_birth, live, live_logL, live_logL_birth = ns_sim()

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