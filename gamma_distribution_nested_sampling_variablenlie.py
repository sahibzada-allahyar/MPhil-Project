# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 09:45:23 2022

@author: Allahyar
"""

from scipy.stats import powerlaw
from scipy.stats import gamma
import matplotlib.pyplot as plt
import numpy as np
import sys 
from scipy.special import logsumexp

nlive =50#NB skilling has nlive=1 in his plots for emphasis
ndead = 10000
C = 10
sigma = 0.01
#k =nlive



# 'PolyChord' parameters
nlive = 500
nprior = nlive*10
ndead = nlive*35

# Generate a dynamic set of live points

# PolyChord cosmo run
nbirth = [nprior] + [-1] * nlive*9 + [0] * ndead + [-1] * (nlive-1)

# Simple run with constant nlive
#nbirth = [nlive] + [0] * (ndead-1)
nlive = np.cumsum(nbirth)




#bring this one back from dead for gaussian L
def loglikelihood(X):
     return -X**(2/C)/(2*sigma**2)
def X(logL_):
    return np.exp(-logL_)
  

 
#power law method

def gen_ns_run(nlive):
     t = powerlaw(nlive).rvs()
     logX = np.log(t).cumsum()
     logL = loglikelihood(np.exp(logX))
     return logX, logL



logXreal, logL = gen_ns_run(nlive)

def logX_powerlaw(nlive):
    t = powerlaw(nlive).rvs()
    logX = np.log(t).cumsum()
    return logX



def logX_gamma(logL, nlive, k):
    ndead = len(logL)    
    y = nlive[1:] * (logL[1:]-logL[:-1])
    cumsumy = np.cumsum(y)
    rollingsumy = np.concatenate((cumsumy[k-1],cumsumy[k:] - cumsumy[:-k]),axis=None)[:-1]
    theta = (logL[(k//2):len(logL)-(k//2)-1] - logL[k//2-1:len(logL)-(k//2)-2]) / rollingsumy
    logt = - theta * gamma(a=k).rvs(len(theta))
    logX = logt.cumsum()
    return logX


def logX_gamma_append(logL, nlive, k):
    ndead = len(logL)    
    y = nlive[1:] * (logL[1:]-logL[:-1])
    cumsumy = np.cumsum(y)
    rollingsumy = np.concatenate((cumsumy[k-1],cumsumy[k:] - cumsumy[:-k]),axis=None)
    theta = (logL[(k//2):len(logL)-(k//2)] - logL[k//2-1:len(logL)-(k//2)-1]) / rollingsumy
    logt = - theta * gamma(a=k).rvs(len(theta))
    logXg = logt.cumsum()
    t = np.concatenate((powerlaw(nlive[:k//2]).rvs(),powerlaw(nlive[len(nlive)-k//2:]).rvs()),axis=None)
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

sys.exit(0)



k=100

logZ_all1=np.array([])
logZ_all2=np.array([])
for m in range(1):
    logXreal, logL = gen_ns_run(nlive)
    plt.axvline(logZ(logL,logXreal),color='k')
    logZ_pl=np.ones(1000)
    for _ in range(1000):
        logZ_pl[_]=logZ(logL, logX_gamma_append(logL,nlive,k))
    logZ_all1=np.concatenate((logZ_all1,logZ_pl),axis=None)
    logZ_pl=np.ones(1000)
    for _ in range(1000):
        logZ_pl[_]=logZ(logL, logX_powerlaw(nlive))
    logZ_all2=np.concatenate((logZ_all2,logZ_pl),axis=None)
plt.hist(logZ_all1, alpha=0.5)
error_sum1= np.std(logZ_all1)
print('error is on logZ of gamma append run is',error_sum1)
plt.hist(logZ_all2, alpha=0.5)
error_sum2= np.std(logZ_all2)
print('error is on logZ of power run is',error_sum2)
plt.subplots()




import os
import numpy as np
import matplotlib.pyplot as plt


if not os.path.isdir('data.1908.09139'):
     import requests
     import tarfile
     import io

     url = 'https://zenodo.org/record/3371152/files/data.1908.09139.tar.gz?download=1' 
     r = requests.get(url, stream=True)

     file = tarfile.open(fileobj=r.raw, mode="r|gz")
     file.extractall(path=".")

from anesthetic import NestedSamples
planck = NestedSamples(root='./data.1908.09139/klcdm/chains/planck_lensing', label=r'Planck')

logL = planck.logL.to_numpy()
nlive = planck.nlive.to_numpy()
    
    plt.plot(logL,logX_powerlaw(nlive))