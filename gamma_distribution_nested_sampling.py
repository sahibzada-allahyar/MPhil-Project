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

nlive =500#NB skilling has nlive=1 in his plots for emphasis
ndead = 10000
C = 10
sigma = 0.01
#k =nlive


#bring this one back from dead for gaussian L
def loglikelihood(X):
     return -X**(2/C)/(2*sigma**2)

#def loglikelihood(logX):
#     return -np.exp(logX*(2/C)/(2*sigma**2))


#bring this one back from dead for gaussian L
#def X(logL_):
 #   return (-2*logL_*sigma**2)**(C/2)
  
 
#def loglikelihood(X):
 #   return -np.log(X)
# 
##gradient increasing as logX increases
#def loglikelihood(X):
#    return (np.log(X))**2

#
##gradient decreasing as logX increases
#def loglikelihood(X):
#    return 1/(np.log(X))


def X(logL_):
    return np.exp(-logL_)
  

 
# power law method

def gen_ns_run(nlive, ndead):
     t = powerlaw(nlive).rvs(ndead)
     logX = np.log(t).cumsum()
     logL = loglikelihood(np.exp(logX))
     return logX, logL


#fig, ax = plt.subplots()
#for _ in range(3):
#    _, logL = gen_ns_run(nlive,ndead)
#    i = np.arange(ndead)+1
#    logX = -i/nlive
#    plt.plot(logX, np.exp(logL),'.')


logXreal, logL = gen_ns_run(nlive,ndead)

def logX_powerlaw(logL):
    ndead = len(logL)
    t = powerlaw(nlive).rvs(ndead)
    logX = np.log(t).cumsum()
    return logX


#fig, ax = plt.subplots()
#for _ in range(10):
#    plt.plot(logX_powerlaw(logL),np.exp(logL),'C0-')
#
#for _ in range(10):
#    plt.plot(logX_gamma(logL, nlive, k),np.exp(logL[:-k-1]),'C1-')
#
#i = np.arange(ndead)+1
#logX = -i/nlive
#plt.plot(logX, np.exp(loglikelihood(np.exp(logX))),'k-')



#def logX_gamma(logL, nlive, k):
#    ndead = len(logL)
#    logt = np.zeros(ndead-k)
#    #for j in range(ndead-k-2):
#        #theta=(logL[j+1]-logL[j])/(nlive*(logL[j+k]-logL[j]))
#        #logt[j]=-gamma(a=k,scale=theta).rvs(1)
#    theta = (logL[1:ndead-k] - logL[0:ndead-k-1]) / (nlive*(logL[k:ndead-1]-logL[0:ndead-k-1]))
#    logt = - theta * gamma(a=k).rvs(len(theta))
#    logX = logt.cumsum()
#    return logX

#def logX_gamma(logL, nlive, k):
#    ndead = len(logL)
#    #logt = np.zeros(ndead-k)
#    #for j in range(ndead-k-2):
#        #theta=(logL[j+1]-logL[j])/(nlive*(logL[j+k]-logL[j]))
#        #logt[j]=-gamma(a=k,scale=theta).rvs(1)
#    theta = (logL[1+(k//2):ndead-(k//2)] - logL[k//2:ndead-(k//2)-1]) / (nlive*(logL[k:ndead-1]-logL[0:ndead-(k)-1]))
#    logt = - theta * gamma(a=k).rvs(len(theta))
#    logX = logt.cumsum()
#    return logX

def logX_gamma(logL, nlive, k):
    ndead = len(logL)
    #logt = np.zeros(ndead-k)
    #for j in range(ndead-k-2):
        #theta=(logL[j+1]-logL[j])/(nlive*(logL[j+k]-logL[j]))
        #logt[j]=-gamma(a=k,scale=theta).rvs(1)
    theta = (logL[(k//2):ndead-(k//2)-1] - logL[k//2-1:ndead-(k//2)-2]) / (nlive*(logL[k:ndead-1]-logL[0:ndead-(k)-1]))
    logt = - theta * gamma(a=k).rvs(len(theta))
    logX = logt.cumsum()
    return logX


def logX_gamma2(logL, nlive, k):
    ndead = len(logL)
    #logt = np.zeros(ndead-k)
    #for j in range(ndead-k-2):
        #theta=(logL[j+1]-logL[j])/(nlive*(logL[j+k]-logL[j]))
        #logt[j]=-gamma(a=k,scale=theta).rvs(1)
    theta = (logL[(k//2)+1:ndead-(k//2)] - logL[k//2:ndead-(k//2)-1]) / (nlive*(logL[k:ndead-1]-logL[0:ndead-(k)-1]))
    logt = - theta * gamma(a=k).rvs(len(theta))
    logX = logt.cumsum()
    return logX


def logX_gamma_append(logL, nlive, k):
    ndead = len(logL)
    #logt = np.zeros(ndead-k)
    #for j in range(ndead-k-2):
        #theta=(logL[j+1]-logL[j])/(nlive*(logL[j+k]-logL[j]))
        #logt[j]=-gamma(a=k,scale=theta).rvs(1)
    theta = (logL[(k//2):ndead-(k//2)] - logL[k//2-1:ndead-(k//2)-1]) / (nlive*(logL[k:ndead-1]-logL[0:ndead-(k)-1]))
    theta
    logt = - theta * gamma(a=k).rvs(len(theta))
    logXg = logt.cumsum()
    #logX = np.concatenate((logXp[:k//2-1],logXg,logXp[ndead-(k//2)-2:]),axis=None)
#    logX = np.concatenate((logXp[:k//2-1],logXg,logXp[ndead-(k//2)-2:]),axis=None)
#    while logX[k//2-1]-logX[k//2-2]>0 or logX[ndead-k//2-2]-logX[ndead-k//2-3]>0:
#        logXp = logX_powerlaw(logL)
#        logX = np.concatenate((logXp[:k//2-1],logXg,logXp[ndead-(k//2)-2:]),axis=None)
    t = powerlaw(nlive).rvs(k+1)
    logXp = np.flip(np.log(t[:k//2-1]).cumsum())
    logXp = -logXp+logXg[0]
    logXp2 = np.log(t[k//2-1:]).cumsum()
    logXp2= logXg[ndead-k-2]+logXp2
    logX = np.concatenate((logXp,logXg,logXp2),axis=None)
    #while logX[k//2-1]-logX[k//2-2]>0:
    #    logXp = logX_powerlaw(logL)
     #   logX = np.concatenate((logXp[:k//2-1],logXg),axis=None)
    return logX

def logZ(logL,logX):
    #Zm= 0.5*(np.exp(logL[1:ndead])+np.exp(logL[0:ndead-1]))*(np.exp(logX[1:ndead])-np.exp(logX[0:ndead-1]))
    logsum_L=logsumexp([logL[1:],logL[:-1]],axis=0)
    logdiff_X=logsumexp([logX[1:],logX[:-1]],axis=0,b=np.array([-1,1])[:,None])
    logQ=logsum_L+logdiff_X-np.log(2)
    logZ=logsumexp(logQ)
    return logZ


#def logZ_looped(logL,logX):
#    Z=0
#    for i in range(ndead-1):
#        Z=Z+0.5*(np.exp(logL[i+1])+np.exp(logL[i]))*(np.exp(logX[i])-np.exp(logX[i+1]))
#    return np.log(Z)

print(logXreal)
print("the evidence is",logZ(logL,logXreal))

logZ_all=np.array([])
logZ_allreal=np.array([])
for _ in range(250):
    logXreal, logL = gen_ns_run(nlive,ndead)
    plt.axvline(logZ(logL,logXreal),color='k')
    logZ_pl=np.ones(1000)
    logZactual=np.ones(1000)
    for _ in range(1000):
        logZ_pl[_]=logZ(logL, logX_powerlaw(logL))
        logZactual[_]=logZ(logL,logXreal)
    plt.hist([logZ_pl], alpha=0.5)
    logZ_all=np.concatenate((logZ_all,logZ_pl),axis=None)
    logZ_allreal=np.concatenate((logZ_allreal,logZactual),axis=None)
logZ_error= (logZ_all-logZ_allreal)**2
error_sum= ((np.sum(logZ_error))**0.5)/(len(logZ_error))
print('error is on logZ of powerlaw run is',error_sum)
plt.subplots()

#
#
#k=6
#
#logZ_all2=np.array([])
#logZ_allreal=np.array([])
#for _ in range(5):
#    logXreal, logL = gen_ns_run(nlive,ndead)
#    plt.axvline(logZ(logL,logXreal),color='k')
#    logZ_pl=np.ones(1000)
#    logZactual=np.ones(1000)
#    for _ in range(1000):
#        logZ_pl[_]=logZ(logL, logX_gamma_append(logL,nlive,k))
#        logZactual[_]=logZ(logL,logXreal)
#  #  print(logZ_pl)
#    plt.hist([logZ_pl], alpha=0.5)
#    logZ_all2=np.concatenate((logZ_all2,logZ_pl),axis=None)
#    logZ_allreal=np.concatenate((logZ_allreal,logZactual),axis=None)
#logZ_error= (logZ_all2-logZ_allreal)**2
#error_sum= ((np.sum(logZ_error))**0.5)/(len(logZ_error))
#print('error is on logZ of gamma append run is',error_sum)
#plt.subplots()
#
    
k= 6
logZ_all3=np.array([])
logZ_allreal=np.array([])
for _ in range(250):
    logXreal, logL = gen_ns_run(nlive,ndead)
    plt.axvline(logZ(logL,logXreal),color='k')
    logZ_pl=np.ones(1000)
    logZactual=np.ones(1000)
    for _ in range(1000):
        logZ_pl[_]=logZ(logL[(k//(2))+1:ndead-(k//2)], logX_gamma(logL,nlive,k))
        logZactual[_]=logZ(logL,logXreal)
    plt.hist([logZ_pl], alpha=0.5)
    logZ_all3=np.concatenate((logZ_all3,logZ_pl),axis=None)
    logZ_allreal=np.concatenate((logZ_allreal,logZactual),axis=None)
logZ_error= (logZ_all3-logZ_allreal)**2
error_sum= ((np.sum(logZ_error))**0.5)/(len(logZ_error))
print('error is on logZ of gamma run is',error_sum)
plt.subplots()


logZ_all3=np.array([])
logZ_allreal=np.array([])
for _ in range(250):
    logXreal, logL = gen_ns_run(nlive,ndead)
    plt.axvline(logZ(logL,logXreal),color='k')
    logZ_pl=np.ones(1000)
    logZactual=np.ones(1000)
    for _ in range(1000):
        logZ_pl[_]=logZ(logL[(k//(2))-1:ndead-(k//2)-2], logX_gamma(logL,nlive,k))
        logZactual[_]=logZ(logL,logXreal)
    plt.hist([logZ_pl], alpha=0.5)
    logZ_all3=np.concatenate((logZ_all3,logZ_pl),axis=None)
    logZ_allreal=np.concatenate((logZ_allreal,logZactual),axis=None)
logZ_error= (logZ_all3-logZ_allreal)**2
error_sum= ((np.sum(logZ_error))**0.5)/(len(logZ_error))
print('error is on logZ of gamma2 run is',error_sum,'using k value as',k)
plt.subplots()






#k = 6
#for _ in range(10):
#    logXreal, logL = gen_ns_run(nlive,ndead)
#    plt.axvline(logZ(logL,logXreal),color='k')
#    plt.hist([logZ(logL, logX_gamma_append(logL,nlive,k)) for _ in range(1000)], alpha=0.5)
# 
#
#for _ in range(10):
#    logXreal, logL = gen_ns_run(nlive,ndead)
#    plt.axvline(logZ(logL,logXreal),color='k')
#    plt.hist([logZ(logL[k//2-1:-k//2-2], logX_gamma(logL,nlive,k)) for _ in range(1000)], alpha=0.5)
#
#
#plt.subplots()
#
#for _ in range(10):
#    logXreal, logL = gen_ns_run(nlive,ndead)
#    plt.axvline(logZ(logL,logXreal),color='k')
#    plt.hist([logZ(logL[(k//(2))-1:ndead-(k//2)-2], logX_gamma(logL,nlive,k)) for _ in range(1000)], alpha=0.5)





sys.exit(0)


fig, ax = plt.subplots()
i = np.arange(ndead)+1
logXreal1=logXreal[0:ndead]
#logX = -i/nlive
for _ in range(10):
    #plt.plot(logXreal1,logX_powerlaw(logL)-logXreal1,'C0-')
    plt.plot(evidence(logL,logX_powerlaw(logL))-evidence(logL,logXreal1),'C0-')
    
k = 6
i = np.arange(ndead-k-1)+1
logXreal2=logXreal[0:ndead-k-1]
#logX = -i/nlive
for _ in range(10):
    #plt.plot(logXreal2,logX_gamma(logL,nlive,k)-logXreal2,'C1')
    plt.plot(evidence(logL,logX_gamma(logL,nlive,k))-evidence(logL,logXreal2),'C1-')
#
#k = 10
#i = np.arange(ndead-k-1)+1
#logXreal3=logXreal[0:ndead-k-1]
##logX = -i/nlive
#for _ in range(10):
#    plt.plot(logXreal3,logX_gamma(logL,nlive,k)-logXreal3,'C2')
#
#
#k = nlive
#i = np.arange(ndead-k-1)+1
#logXreal3=logXreal[0:ndead-k-1]
##logX = -i/nlive
#for _ in range(10):
#    plt.plot(logXreal3,logX_gamma(logL,nlive,k)-logXreal3,'C3')
#
#
#k = nlive*5
#i = np.arange(ndead-k-1)+1
#logXreal3=logXreal[0:ndead-k-1]
##logX = -i/nlive
#for _ in range(10):
#    plt.plot(logXreal3,logX_gamma(logL,nlive,k)-logXreal3,'C4')
#
#

sys.exit(0)
import tqdm
for j, k in tqdm.tqdm(enumerate([1,5,10,20])):
    i = np.arange(ndead-k-1)+1
    logX = -i/nlive
    for _ in range(3):
        plt.plot(logX,logX_gamma(logL,nlive,k)-logX,'C%i-' % j, label='k=%i' % k)
plt.legend()

#k = nlive//2
i = np.arange(ndead-k-1)+1
logX = -i/nlive
for _ in range(10):
    plt.plot(logX,logX_gamma(logL,nlive,k)-logX,'C2-')

k = 1
i = np.arange(ndead-k-1)+1
logX = -i/nlive
for _ in range(10):
    plt.plot(logX,logX_gamma(logL,nlive,k)-logX,'C2-')


# Figure 5
#fig, ax = plt.subplots()
logX = np.linspace(-70,0,10000)
logL = loglikelihood(np.exp(logX))
#ax.plot(logX, logL)

# Figure 6
#fig, ax = plt.subplots()
t = powerlaw(nlive).rvs(ndead)
logX = np.log(t).cumsum()
logL = loglikelihood(np.exp(logX))
#ax.plot(logX, np.exp(logL),'.')

# Figure 7 & 8
fig, ax = plt.subplots()
logXmatrix1=np.ones(t)
for _ in range(t):
     t = powerlaw(nlive).rvs(ndead)
     logX = np.log(t).cumsum()
     logL = loglikelihood(np.exp(logX))
     realXval=X(logL)
     errors=((np.log(realXval)-logX)**2)
     i = np.arange(ndead)+1
     logX = -i/nlive
    # ax.plot(logX, np.exp(logL),'.')
logt= np.log(t)
theta=np.ones(ndead)
logt_Expectation=np.ones(ndead)/nlive
print(logt_Expectation)
for j in range(ndead-k):
    #the if statement is to make sure we dont divide by zero.
    #if the denominator term in theta is too close to zero, I just let
    #the logt remain the same as it would be for the powerlaw run
    #if (np.log(L[j+k])-np.log(L[j]))>1e-300:
    theta[j]=(logL[j+1]-logL[j])/(nlive*(logL[j+k]-logL[j]))
    logt[j]=-gamma(a=k,scale=theta[j]).rvs(1)
    logt_Expectation[j]= -k*theta[j]
logX = logt.cumsum()
#logL = loglikelihood(np.exp(logX))
#logX=logt_Expectation.cumsum()
ax.plot(logX, np.exp(logL),'.')
logX = np.linspace(-70,0,10000)
logL = loglikelihood(np.exp(logX))
ax.plot(logX, np.exp(logL))
