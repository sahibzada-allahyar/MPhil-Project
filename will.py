from scipy.stats import powerlaw
from scipy.stats import gamma
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
import anesthetic

nlive =50
ndead = 10000
C = 10
sigma = 0.01


def loglikelihood(X):
     return -X**(2/C)/(2*sigma**2)


def gen_ns_run(nlive, ndead):
     t = powerlaw(nlive).rvs(ndead)
     logX = np.log(t).cumsum()
     logL = loglikelihood(np.exp(logX))
     return logX, logL


def logX_powerlaw(logL,nlive, n=1):
    ndead = len(logL)
    t = powerlaw(nlive).rvs((n, ndead))
    logX = np.log(t).cumsum(axis=-1)
    return np.squeeze(logX)

def logX_powerlaw2(logL,nlive, n=1):
    ndead = len(logL)
    logt = np.random.normal(-1/nlive, 1/nlive, (n, ndead))
    logX = logt.cumsum(axis=-1)
    return np.squeeze(logX)

def logX_gamma(logL, nlive, k, n=1, s=1):
    m = len(logL)
    theta = (logL[k:-k-1] - logL[k+1:m-k])/(nlive*(logL[:-2*k-1] - logL[2*k+1:]))
    logt = - theta * gamma(a=2*k+1).rvs((n, len(theta)))
    logt = np.concatenate([np.log(powerlaw(nlive).rvs((n,k+s))),logt,np.log(powerlaw(nlive).rvs((n,k+(1-s))))],axis=-1)
    logX = logt.cumsum(axis=-1)
    return np.squeeze(logX)

def logX_gamma2(logL, nlive, k, n=1):
    m = len(logL)
    theta = (logL[k:-k-1] - logL[k+1:m-k])/((logL[:-2*k-1] - logL[2*k+1:])/(2*k+1))
    logt = theta * np.random.normal(-1/nlive, 1/nlive / (2*k+1)**0.5, (n, len(theta)))
    #logt = theta * np.random.normal(-1/nlive, 1/nlive, (n, len(theta)))
    logt = np.concatenate([np.log(powerlaw(nlive).rvs((n,k))),logt,np.log(powerlaw(nlive).rvs((n,k+1)))],axis=-1)
    logX = logt.cumsum(axis=-1)
    return np.squeeze(logX)



def logsubexp(a,b):
    return np.log(np.exp(a-b)-1)+b


def logZ(logL,logX):
    logsum_L = logsumexp([logL[1:],logL[:-1]],axis=0)
    b = [-np.ones_like(logX[...,1:]),np.ones_like(logX[...,:-1])] 
    logdiff_X, sgn = logsumexp([logX[...,1:],logX[...,:-1]],axis=0,b=b, return_sign=True)
    logQ=logsum_L+logdiff_X-np.log(2)
    logZ=logsumexp(logQ,b=sgn,axis=-1)
    return logZ



k = 10
n = 1000

logXreal, logL = gen_ns_run(nlive,ndead)
#logX = logX_powerlaw2(logL, nlive,n)
#plt.hist(logZ(logL, logX))
logX = logX_powerlaw(logL, nlive,n)
plt.hist(logZ(logL, logX))

#logX = logX_gamma2(logL, nlive,k, n)
#plt.hist(logZ(logL, logX))
logX = logX_gamma(logL, nlive, k, n)
plt.hist(logZ(logL, logX))
plt.axvline(logZ(logL,logXreal),color='k')

logX
logL
k=10
plt.subplots()
plt.plot(logXreal,logL,'o-')
plt.plot(logX_powerlaw(logL, nlive),logL,'o-')
plt.plot(logX_gamma(logL, nlive, k, s=1),logL,'o-',)

plt.subplots()
#logXreal, logL = gen_ns_run(nlive,ndead)
for _ in range(100):
    plt.plot(logX_powerlaw(logL, nlive) - logXreal,'C1')
    plt.plot(logX_gamma(logL, nlive, 3) - logXreal,'C2')

logX

for k in [1,3,5,10,30,50,100]:
    logX = logX_gamma(logL, nlive, k, n)
    plt.hist(logZ(logL, logX))
plt.axvline(logZ(logL,logXreal),color='k')

k=0
logXreal, logL = gen_ns_run(nlive, ndead)
logX = logX_powerlaw(logL, nlive, n)

plt.hist(logZ(logL, logX))
logX = logX_gamma(logL, nlive, k, n)
plt.hist(logZ(logL, logX))
plt.axvline(logZ(logL,logXreal),color='k')

logX.shape
logZ(logL, logX)

g= -np.exp(-(2*logXreal)/C)*(C*sigma**2)

k = 5
logL
k = 5
for k in [1,3,5,10,30,50,100]:
    approx_g = -(logL[2*k:] - logL[:-2*k])/((2*k)/nlive)
    plt.plot(logXreal[k:-k], (g[k:-k]- 1/approx_g)/g[k:-k])


plt.plot(logXreal[k:-k], (g[k:-k]- 1/approx_g)/g[k:-k])
plt.plot(logXreal, g)

for k in [1,3,5,10,30,50,100]:
    approx_g = -(logL[2*k:] - logL[:-2*k])/((2*k)/nlive)
    plt.plot(logXreal[k:-k], 1/approx_g)

plt.plot(logXreal[k:-k], g[k:-k])

k = nlive
ndead = 10000
logXreal, logL = gen_ns_run(nlive,ndead)
plt.plot(
        (logL[k:-k-1] - logL[k+1:-k])/((logL[:-2*k-1] - logL[2*k+1:])/(2*k+1))
        )
plt.plot(
        (logXreal[k:-k-1] - logXreal[k+1:-k])/((logXreal[:-2*k-1] - logXreal[2*k+1:])/(2*k+1))
        )

logL[2*


