#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 13:56:59 2022

@author: sahibzadaallahyar
"""


from scipy.stats import powerlaw
from scipy.stats import gamma
import matplotlib.pyplot as plt
import numpy as np
import sys 
from scipy.special import logsumexp
import anesthetic

nlive =50#NB skilling has nlive=1 in his plots for emphasis
ndead = 10000
C = 10
sigma = 0.01
#k =nlive


#bring this one back from dead for gaussian L
def loglikelihood(X):
     return -X**(2/C)/(2*sigma**2)

 
# power law method

def gen_ns_run(nlive, ndead):
     t = powerlaw(nlive).rvs(ndead)
     logX = np.log(t).cumsum()
     logL = loglikelihood(np.exp(logX))
     return logX, logL



logXreal, logL = gen_ns_run(nlive,ndead)

def logX_powerlaw(logL):
    ndead = len(logL)
    t = powerlaw(nlive).rvs(ndead)
    logX = np.log(t).cumsum()
    return logX

def logX_gamma(logL, nlive, k, n=1):
    ndead = len(logL)
    theta = (logL[(k//2):ndead-(k//2)-1] - logL[k//2-1:ndead-(k//2)-2]) / (nlive*(logL[k:ndead-1]-logL[0:ndead-(k)-1]))
    logt = - theta * gamma(a=k).rvs((n, len(theta)))
    logX = logt.cumsum(axis=-1)
    return np.squeeze(logX)


def logX_gamma2(logL, nlive, k, n=1):
    ndead = len(logL)
    theta = (logL[(k//2)+1:ndead-(k//2)] - logL[k//2:ndead-(k//2)-1]) / (nlive*(logL[k:ndead-1]-logL[0:ndead-(k)-1]))
    logt = - theta * gamma(a=k).rvs((n,len(theta)))
    logX = logt.cumsum(axis=-1)
    return np.squeeze(logX)


def logX_gamma_append(logL, nlive, k):
    ndead = len(logL)
    theta = (logL[(k//2):ndead-(k//2)-1] - logL[k//2-1:ndead-(k//2)-2]) / (nlive*(logL[k:ndead-1]-logL[0:ndead-(k)-1]))
    theta
    logt = - theta * gamma(a=k).rvs(len(theta))
    logXg = logt.cumsum()
    t = powerlaw(nlive).rvs(k+1)
    logXp = np.flip(np.log(t[:k//2-1]).cumsum())
    logXp = -logXp+logXg[0]
    logXp2 = np.log(t[k//2-1:]).cumsum()
    logXp2= logXg[ndead-k-2]+logXp2
    logX = np.concatenate((logXp,logXg,logXp2),axis=None)
    return logX

def logsubexp(a,b):
    return np.log(np.exp(a-b)-1)+b

def logZ(logL,logX):
    logsum_L=logsumexp([logL[1:],logL[:-1]],axis=0)
    logdiff_X=logsubexp(logX[...,:-1],logX[...,1:])
    logQ=logsum_L+logdiff_X-np.log(2)
    logZ=logsumexp(logQ, axis=-1)
    return logZ


print(logXreal)
print("the evidence is",logZ(logL,logXreal))
#our program for some reason likes logZ=-37.79847

colors = ["red", "blue" , "green", "orange", "purple"]
import tqdm
for m in tqdm.trange(5):
    logZ_all3=np.array([])
    kval=np.array([])
    logZ_allreal=np.array([])
    stds=np.array([])
    logXreal, logL = gen_ns_run(nlive,ndead)
    plt.axhline(logZ(logL,logXreal),color='k')
    logZ_pl=np.ones(1000)
    for _ in range(1000):
        logZ_pl[_]=logZ(logL, logX_powerlaw(logL))

    logZ_all3=np.concatenate((logZ_all3,np.mean(logZ_pl)),axis=None)
    stds= np.concatenate((stds,np.std(logZ_pl)),axis=None)
    kval=np.concatenate((kval,1),axis=None)
    for k in tqdm.tqdm([ 6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34,
           36, 38, 40, 42, 44, 46, 48, 50]):
        logZ_pl=logZ(logL[(k//(2))-1:ndead-(k//2)-2], logX_gamma(logL,nlive,k,1000))
        logZ_all3=np.concatenate((logZ_all3,np.mean(logZ_pl)),axis=None)
        stds= np.concatenate((stds,np.std(logZ_pl)),axis=None)
        kval=np.concatenate((kval,k),axis=None)
    plt.errorbar(kval,logZ_all3,yerr=stds,color=colors[m])
    print('k values',kval,'logZ vals',logZ_all3)
    
plt.subplots()


sys.exit(0)
