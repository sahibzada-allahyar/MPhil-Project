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


def X(logL_):
    return np.exp(-logL_)
  

 
# power law method

def gen_ns_run(nlive, ndead):
     t = powerlaw(nlive).rvs(ndead)
     logX = np.log(t).cumsum()
     logL = loglikelihood(np.exp(logX))
     return logX, logL



logXreal, logL = gen_ns_run(nlive,ndead)

def logX_powerlaw(logL,nlive):
    ndead = len(logL)
    t = powerlaw(nlive).rvs(ndead)
    logX = np.log(t).cumsum()
    return logX

def logX_gamma(logL, nlive, k):
    ndead = len(logL)
    theta = (logL[(k//2):ndead-(k//2)-1] - logL[k//2-1:ndead-(k//2)-2]) / (nlive*(logL[k:ndead-1]-logL[0:ndead-(k)-1]))
    logt = - theta * gamma(a=k).rvs(len(theta))
    logX = logt.cumsum()
    return logX


def logX_gamma2(logL, nlive, k):
    ndead = len(logL)
    theta = (logL[(k//2)+1:ndead-(k//2)] - logL[k//2:ndead-(k//2)-1]) / (nlive*(logL[k:ndead-1]-logL[0:ndead-(k)-1]))
    logt = - theta * gamma(a=k).rvs(len(theta))
    logX = logt.cumsum()
    return logX


def logX_gamma_append(logL, nlive, k):
    ndead = len(logL)
    theta = (logL[(k//2):ndead-(k//2)] - logL[k//2-1:ndead-(k//2)-1]) / (nlive*(logL[k:ndead]-logL[0:ndead-(k)]))
    logt = - theta * gamma(a=k).rvs(len(theta))
    logXg = logt.cumsum()
    t = powerlaw(nlive).rvs(k)
    logXp = np.log(t[:k//2]).cumsum()
    logXg = logXp[k//2-1]+logXg
    logXp2 = np.log(t[k//2:]).cumsum()
    logXp2= logXg[ndead-k-1]+logXp2
    logX = np.concatenate((logXp,logXg,logXp2),axis=None)
    return logX

def logZ(logL,logX):
    logsum_L=logsumexp([logL[1:],logL[:-1]],axis=0)
    logdiff_X=logsumexp([logX[1:],logX[:-1]],axis=0,b=np.array([-1,1])[:,None])
    logQ=logsum_L+logdiff_X-np.log(2)
    logZ=logsumexp(logQ)
    return logZ


print(logXreal)
print("the evidence is",logZ(logL,logXreal))
#our program for some reason likes logZ=-37.79847



colors = ["red", "blue" , "green", "orange", "purple"]
for m in range(5):
    logZ_all3=np.array([])
    kval=np.array([])
    logZ_allreal=np.array([])
    stds=np.array([])
    logXreal, logL = gen_ns_run(nlive,ndead)
    plt.axhline(logZ(logL,logXreal),color='k')
    logZ_pl=np.ones(1000)
    for _ in range(1000):
        logZ_pl[_]=logZ(logL, logX_powerlaw(logL,nlive))
    logZ_all3=np.concatenate((logZ_all3,np.mean(logZ_pl)),axis=None)
    stds= np.concatenate((stds,np.std(logZ_pl)),axis=None)
    kval=np.concatenate((kval,1),axis=None)
    for k in [  4, 6,  8, 10]:
        logZ_pl=np.ones(1000)
        for _ in range(1000):
            logZ_pl[_]=logZ(logL, logX_gamma_append(logL,nlive,k))
        logZ_all3=np.concatenate((logZ_all3,np.mean(logZ_pl)),axis=None)
        stds= np.concatenate((stds,np.std(logZ_pl)),axis=None)
        kval=np.concatenate((kval,k),axis=None)
    plt.errorbar(kval,logZ_all3,yerr=stds,color=colors[m])
    print('k values',kval,'logZ vals',logZ_all3)
print('error is on logZ of gamma2 run is',error_sum,'using k value as',k)
plt.subplots()

