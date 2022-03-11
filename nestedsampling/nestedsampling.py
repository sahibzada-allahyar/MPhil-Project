#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 16:19:45 2022

@author: sahibzadaallahyar
"""

class MetropolisNS():
    def __init__(self,loglikelihood,prior_bounds, ndims=2, nlive=125, num_repeats=10):
        """Metropolis Hastings Nested Sampling run"""
        low= prior_bounds[1]
        high=prior_bounds[2]
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
                    if loglikelihood(live_point) > Lmin:
                        break
                live_points[i, :] = live_point
                live_likes[i] = loglikelihood(live_points[i])
        return np.array(dead_points), np.array(dead_likes), np.array(birth_likes), live_points, live_likes, live_birth_likes
    
