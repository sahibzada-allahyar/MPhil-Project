#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 19:28:22 2022

@author: sahibzadaallahyar
"""
import numpy as np 
from anesthetic import MCMCSamples, NestedSamples
import tqdm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


ndims = 3
columns = ['x%i' % i for i in range(ndims)]
tex = {p: '$x_%i$' % i  for i, p in enumerate(columns)}

def loglikelihood(x):
    sigma= 0.1
    x0,x1,x2= x[:]
    
    logL = -np.log(sigma**(3)*(2*np.pi)**(3/2))
    logL -= ((x0-0.5)**2)/(2*sigma**2)
    logL -= ((x1-0.5)**2)/(2*sigma**2)
    logL -= ((x2-0.5)**2)/(2*sigma**2)
    #logL -= ((x3-0.5)**2)/(2*sigma**2)
    return logL

# def loglikelihood(x):
#     mean = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
#     cov = np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.5, 0.0,  0.0, 0.0, 0.0], [0.0, 0.0, 0.9, 0.0, 0.0, 0.0],[ 0.0, 0.0, 0.0, 0.9,0.0,0.0],[0.0, 0.0, 0.0, 0.0, 0.9,0.0],[0.0, 0.0, 0.0, 0.0, 0.0,0.9]])
#     logL = multivariate_normal.logpdf(x, mean=mean, cov=cov)
#     return logL
    

def ns_sim_mh(ndims=3, nlive=125,num_repeats=40):
    """Metropolis Hastings Nested Sampling run"""
    low=(0,0,0)
    high=(1,1,1)
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
        for gg in range(num_repeats):
            while live_likes[i] <= Lmin:
                live_points[i, :] = live_points[i]+np.random.multivariate_normal(mean=np.zeros(len(live_points[0])), cov=np.cov(live_points.T)/2)
                live_likes[i] = loglikelihood(live_points[i])
    return np.array(dead_points), np.array(dead_likes), np.array(birth_likes), live_points, live_likes, live_birth_likes



def ns_sim(ndims=3, nlive=125):
    """Brute force Nested Sampling run"""
    low=(0,0,0)
    high=(1,1,1)
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

MHns = NestedSamples(data=data, columns=columns, logL=logL, logL_birth=logL_birth, tex=tex)
live_ns = NestedSamples(data=live, columns=columns, logL=live_logL, logL_birth=live_logL_birth, tex=tex)
MHns.plot_2d(['x0','x1','x2'])

data, logL, logL_birth, live, live_logL, live_logL_birth = ns_sim()

ns = NestedSamples(data=data, columns=columns, logL=logL, logL_birth=logL_birth, tex=tex)
live_ns = NestedSamples(data=live, columns=columns, logL=live_logL, logL_birth=live_logL_birth, tex=tex)


ns.plot_2d(['x0','x1','x2'])

