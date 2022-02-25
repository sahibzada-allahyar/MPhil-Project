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
import anesthetic
import numpy.polynomial.polynomial as poly

nlive =1000#NB skilling has nlive=1 in his plots for emphasis
ndead = 100000
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


def logX_gamma_append2(logL, nlive, k):
    ndead = len(logL)
    theta = (logL[k+1:-k] - logL[k:-k-1]) / (nlive*(logL[2*k+1:]-logL[0:-2*k-1]))
    logt = - theta * gamma(a=2*k+1).rvs(len(theta))
    logXg = logt.cumsum()
    t = powerlaw(nlive).rvs(2*k+1)
    logXp = np.log(t[:k+1]).cumsum()
    logXg = logXp[k]+logXg
    logXp2 = np.log(t[k+1:]).cumsum()
    logXp2= logXg[ndead-2*k-2]+logXp2
    logX = np.concatenate((logXp,logXg,logXp2),axis=None)
    return logX



def logX_gamma_rolling(logL, nlive, k):
    ndead = len(logL)
    logXpl1= logX_powerlaw(logL, nlive)
    logXpl2= logX_powerlaw(logL, nlive)
    logXpl3= logX_powerlaw(logL, nlive)
    logXpl4= logX_powerlaw(logL, nlive)
    logXpl5= logX_powerlaw(logL, nlive)
    logXpl6= logX_powerlaw(logL, nlive)
    logXpl7= logX_powerlaw(logL, nlive)
    logXpl8= logX_powerlaw(logL, nlive)
    logXpl9= logX_powerlaw(logL, nlive)
    logXpl10= logX_powerlaw(logL, nlive)
    logXpl11= logX_powerlaw(logL, nlive)
    logXpl12= logX_powerlaw(logL, nlive)
    logXpl13= logX_powerlaw(logL, nlive)
    logXpl14= logX_powerlaw(logL, nlive)
    logXpl15= logX_powerlaw(logL, nlive)
    logXpl16= logX_powerlaw(logL, nlive)
    logXpl17= logX_powerlaw(logL, nlive)
    logXpl18= logX_powerlaw(logL, nlive)
    logXpl19= logX_powerlaw(logL, nlive)
    logXpl20= logX_powerlaw(logL, nlive)
    logXpl21= logX_powerlaw(logL, nlive)
    logXpl22= logX_powerlaw(logL, nlive)
    logXpl23= logX_powerlaw(logL, nlive)
    logXpl24= logX_powerlaw(logL, nlive)
    logXpl25= logX_powerlaw(logL, nlive)
    logXpl26= logX_powerlaw(logL, nlive)
    ol=(logXpl1+logXpl2+logXpl3+logXpl4+logXpl5+logXpl6+logXpl7+logXpl8+logXpl9+logXpl10+logXpl11+logXpl12+logXpl13+logXpl14+logXpl15+logXpl16+logXpl17+logXpl18+logXpl19 +logXpl20 +logXpl21 +logXpl22 +logXpl23 +logXpl24 +logXpl25 +logXpl26 )
    ol= ol/26
    cumsumg=(-(2*k+1) / (nlive*(logL[2*k+1:]-logL[0:-2*k-1]))).cumsum()
    b=2*k
    rollingsumg= np.concatenate((cumsumg[b]/(b+1),(cumsumg[b+1:]-cumsumg[:-b-1])/(b+1)),axis=None)
    g= -np.exp(-(2*ol)/C)*(C*sigma**2)
    # plt.plot(g[k+1+b//2:len(ol)-b//2-k],ol[k+1+b//2:len(ol)-b//2-k])
    # plt.plot(rollingsumg,ol[k+1+b//2:len(ol)-b//2-k])
    theta = (logL[k+1+b//2:-b//2-k] - logL[k+b//2:-b//2-k-1])*rollingsumg/(2*k+1)
    logt = theta * gamma(a=2*k+1).rvs(len(theta))
    logXg = logt.cumsum()
    t = powerlaw(nlive).rvs(2*k+1+b)
    logXp = np.log(t[:k+1+b//2]).cumsum()
    logXg = logXp[k+b//2]+logXg
    logXp2 = np.log(t[k+1+b//2:]).cumsum()
    logXp2= logXg[ndead-2*k-2-b]+logXp2
    logX = np.concatenate((logXp,logXg,logXp2),axis=None)
    return logX




# def logX_gamma_append2(logL, nlive, k):
#     ndead = len(logL)
#     theta = (logL[k+1:-k] - logL[k:-k-1]) / (nlive*(logL[2*k+1:]-logL[k+1:-k] + logL[k:-k-1]-logL[0:-2*k-1]))
#     logt = - theta * gamma(a=2*k).rvs(len(theta))
#     logXg = logt.cumsum()
#     t = powerlaw(nlive).rvs(2*k+1)
#     logXp = np.log(t[:k+1]).cumsum()
#     logXg = logXp[k]+logXg
#     logXp2 = np.log(t[k+1:]).cumsum()
#     logXp2= logXg[ndead-2*k-2]+logXp2
#     logX = np.concatenate((logXp,logXg,logXp2),axis=None)
#     return logX



# def logX_gamma_append3(logL, nlive, k):
#     ndead = len(logL)
#     logt = - ((logL[k+1:-k] - logL[k:-k-1])*(2*k)) / (nlive*(logL[2*k+1:]-logL[k+1:-k] + logL[k:-k-1]-logL[0:-2*k-1]))
#     logXg = logt.cumsum()
#     t = powerlaw(nlive).rvs(2*k+1)
#     logXp = np.log(t[:k+1]).cumsum()
#     logXg = logXp[k]+logXg
#     logXp2 = np.log(t[k+1:]).cumsum()
#     logXp2= logXg[ndead-2*k-2]+logXp2
#     logX = np.concatenate((logXp,logXg,logXp2),axis=None)
#     return logX


# def logX_gamma_append3(logL, nlive, k):
#     ndead = len(logL)
#     logt = - ((logL[(k//2):ndead-(k//2)] - logL[k//2-1:ndead-(k//2)-1])*k) / (nlive*(logL[k:ndead]-logL[0:ndead-(k)]))
#     logXg = logt.cumsum()
#     t = powerlaw(nlive).rvs(k)
#     logXp = np.log(t[:k//2]).cumsum()
#     logXg = logXp[k//2-1]+logXg
#     logXp2 = np.log(t[k//2:]).cumsum()
#     logXp2= logXg[ndead-k-1]+logXp2
#     logX = np.concatenate((logXp,logXg,logXp2),axis=None)
#     return logX


def fitting_run(logL,degree,nlive):
    logX1= logX_powerlaw(logL,nlive)
    coeffs = np.polyfit(logX1,logL,100)
    ffit = np.poly1d(coeffs)
    g= np.vectorize(ffit.deriv())(logX1)   
    g=1/g
    logt= -(logL[0:-1]-logL[1:])*g[1:]
    logt= np.concatenate((np.log((powerlaw(nlive).rvs(1))),logt),axis=None)
    logX= logt.cumsum()
    plt.plot(np.vectorize(ffit)(logX1),logX1)
    plt.plot(logL,logX1)
    return logX


def analytical_grad(logL,logXreal,nlive):
    # g= -np.exp(-(2*logXreal)/C)*(C*sigma**2)
    # logt= -(logL[0:-1]-logL[1:])*g[1:]
    # logt= np.concatenate((np.log((powerlaw(nlive).rvs(1))),logt),axis=None)
    # logX= logt.cumsum()
    g= -np.exp(-(2*logXreal)/C)*(C*sigma**2)
    logt= -(logL[0:-1]-logL[1:])*g[1:]
    logt= np.concatenate((np.log((powerlaw(nlive).rvs(1))),logt),axis=None)
    logX= logt.cumsum()
    return logX



def analytical_grad2(logL,logXreal,nlive,k):
    ndead = len(logL)
    g= -np.exp(-(2*logXreal)/C)*(C*sigma**2)
    theta = -(logL[(k//2):ndead-(k//2)] - logL[k//2-1:ndead-(k//2)-1]) * g[(k//2):ndead-(k//2)] / k
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

k=30
print("the evidence is",logZ(logL,logXreal))
print("the evidence is",logZ(logL,logX_gamma_rolling(logL, nlive, k)))
#our program for some reason likes logZ=-37.79847


sys.exit(0)


k=500
logXreal, logL = gen_ns_run(nlive,ndead)




k=100
logXpl1= logX_powerlaw(logL, nlive)
logXpl2= logX_powerlaw(logL, nlive)
logXpl3= logX_powerlaw(logL, nlive)
logXpl4= logX_powerlaw(logL, nlive)
logXpl5= logX_powerlaw(logL, nlive)
logXpl6= logX_powerlaw(logL, nlive)
logXpl7= logX_powerlaw(logL, nlive)
logXpl8= logX_powerlaw(logL, nlive)
logXpl9= logX_powerlaw(logL, nlive)
logXpl10= logX_powerlaw(logL, nlive)
logXpl11= logX_powerlaw(logL, nlive)
logXpl12= logX_powerlaw(logL, nlive)
logXpl13= logX_powerlaw(logL, nlive)
logXpl14= logX_powerlaw(logL, nlive)
logXpl15= logX_powerlaw(logL, nlive)
logXpl16= logX_powerlaw(logL, nlive)
logXpl17= logX_powerlaw(logL, nlive)
logXpl18= logX_powerlaw(logL, nlive)
logXpl19= logX_powerlaw(logL, nlive)
logXpl20= logX_powerlaw(logL, nlive)
logXpl21= logX_powerlaw(logL, nlive)
logXpl22= logX_powerlaw(logL, nlive)
logXpl23= logX_powerlaw(logL, nlive)
logXpl24= logX_powerlaw(logL, nlive)
logXpl25= logX_powerlaw(logL, nlive)
logXpl26= logX_powerlaw(logL, nlive)
ol=(logXpl1+logXpl2+logXpl3+logXpl4+logXpl5+logXpl6+logXpl7+logXpl8+logXpl9+logXpl10+logXpl11+logXpl12+logXpl13+logXpl14+logXpl15+logXpl16+logXpl17+logXpl18+logXpl19 +logXpl20 +logXpl21 +logXpl22 +logXpl23 +logXpl24 +logXpl25 +logXpl26 )
ol= ol/26
cumsumg=(-(2*k) / (nlive*(logL[2*k+1:]-logL[k+1:-k] + logL[k:-k-1]-logL[0:-2*k-1]))).cumsum()
rollingsumg= (cumsumg[k:]-cumsumg[:-k])/(k+1)
g= -np.exp(-(2*ol)/C)*(C*sigma**2)
plt.plot(g[k+1+k//2:len(ol)-k//2-k],ol[k+1+k//2:len(ol)-k//2-k])
plt.plot(rollingsumg,ol[k+1+k//2:len(ol)-k//2-k])
plt.plot((-(2*k) / (nlive*(logL[2*k+1:]-logL[k+1:-k] + logL[k:-k-1]-logL[0:-2*k-1]))),ol[k+1:-k])



f=np.ones(1000)
f2=np.ones(1000)
for _ in range(1000):
    f[_]=-(2*k) / (nlive*(logL[2*k+1:]-logL[k+1:-k] + logL[k:-k-1]-logL[0:-2*k-1]))
    f2[_]=-(2*k+1) / (nlive*(logL[2*k+1:]-logL[0:-2*k-1]))




colors = ["red", "blue" , "green", "orange", "purple","red", "blue" , "green", "orange", "purple","red", "blue" , "green", "orange", "purple"]
for m in range(5):
    logZ_all3=np.array([])
    kval=np.array([])
    logZ_allreal=np.array([])
    stds=np.array([])
    logXreal, logL = gen_ns_run(nlive,ndead)
    plt.axhline(logZ(logL,logXreal),color='k')
    logZ_pl=np.ones(100)
    for _ in range(100):
        logZ_pl[_]=logZ(logL, logX_powerlaw(logL,nlive))
    logZ_all3=np.concatenate((logZ_all3,np.mean(logZ_pl)),axis=None)
    stds= np.concatenate((stds,np.std(logZ_pl)),axis=None)
    kval=np.concatenate((kval,1),axis=None)
    for k in [5,10,20,50,100,150,200,300,400]:
        logZ_pl=np.ones(100)
        for _ in range(100):
            logZ_pl[_]=logZ(logL, logX_gamma_rolling(logL,nlive,k))
        logZ_all3=np.concatenate((logZ_all3,np.mean(logZ_pl)),axis=None)
        stds= np.concatenate((stds,np.std(logZ_pl)),axis=None)
        kval=np.concatenate((kval,k),axis=None)
    plt.errorbar(kval,logZ_all3,yerr=stds,color=colors[m])
    print('k values',kval,'logZ vals',logZ_all3)
plt.subplots()

colors = ["red", "blue" , "green", "orange", "purple"]
for m in range(1):
    logZ_all3=np.array([])
    kval=np.array([])
    logZ_allreal=np.array([])
    stds=np.array([])
    logXreal, logL = gen_ns_run(nlive,ndead)
    plt.axhline(logZ(logL,logXreal),color='k')
    logZ_pl=np.ones(100)
    for _ in range(100):
        logZ_pl[_]=logZ(logL, logX_powerlaw(logL,nlive))
    logZ_all3=np.concatenate((logZ_all3,np.mean(logZ_pl)),axis=None)
    stds= np.concatenate((stds,np.std(logZ_pl)),axis=None)
    kval=np.concatenate((kval,1),axis=None)
    for k in [ 2,4,6,8]:
        logZ_pl=np.ones(100)
        for _ in range(100):
            logZ_pl[_]=logZ(logL, analytical_grad2(logL,logXreal,nlive,k))
        logZ_all3=np.concatenate((logZ_all3,np.mean(logZ_pl)),axis=None)
        stds= np.concatenate((stds,np.std(logZ_pl)),axis=None)
        kval=np.concatenate((kval,k),axis=None)
    plt.errorbar(kval,logZ_all3,yerr=stds,color=colors[m])
    print('k values',kval,'logZ vals',logZ_all3)
plt.subplots()


sys.exit(0)

colors = ["red", "blue" , "green", "orange", "purple"]
for m in range(5):
    logZ_all3=np.array([])
    kval=np.array([])
    logZ_allreal=np.array([])
    stds=np.array([])
    logXreal, logL = gen_ns_run(nlive,ndead)
    plt.axhline(logZ(logL,logXreal),color='k')
    logZ_pl=np.ones(1000)
    for k in [  4, 6,  8, 10]:
        logZ_pl=np.ones(1000)
        for _ in range(1000):
            logZ_pl[_]=logZ(logL, logX_gamma_append(logL,nlive,k))
        logZ_all3=np.concatenate((logZ_all3,np.mean(logZ_pl)),axis=None)
        stds= np.concatenate((stds,np.std(logZ_pl)),axis=None)
        kval=np.concatenate((kval,k),axis=None)
    plt.errorbar(kval,logZ_all3,yerr=stds,color=colors[m])
    logZ_all3=np.array([])
    kval=np.array([])
    logZ_allreal=np.array([])
    stds=np.array([])
    for k in [  4, 6,  8, 10]:
        logZ_pl=np.ones(1000)
        for _ in range(1000):
            logZ_pl[_]=logZ(logL[(k//(2))-1:ndead-(k//2)-2], logX_gamma(logL,nlive,k))
        logZ_all3=np.concatenate((logZ_all3,np.mean(logZ_pl)),axis=None)
        stds= np.concatenate((stds,np.std(logZ_pl)),axis=None)
        kval=np.concatenate((kval,k),axis=None)
    plt.errorbar(kval,logZ_all3,yerr=stds,color='black')
    logZ_all3=np.array([])
    kval=np.array([])
    logZ_allreal=np.array([])
    stds=np.array([])
    for k in [  4, 6,  8, 10]:
        logZ_pl=np.ones(1000)
        for _ in range(1000):
            logZ_pl[_]=logZ(logL[(k//(2)):ndead-(k//2)-1], logX_gamma(logL,nlive,k))
        logZ_all3=np.concatenate((logZ_all3,np.mean(logZ_pl)),axis=None)
        stds= np.concatenate((stds,np.std(logZ_pl)),axis=None)
        kval=np.concatenate((kval,k),axis=None)
    plt.errorbar(kval,logZ_all3,yerr=stds,color='black')
    print('k values',kval,'logZ vals',logZ_all3)
print('error is on logZ of gamma2 run is',error_sum,'using k value as',k)
plt.subplots()


sys.exit(0)

logZ_all=np.array([])
logZ_allreal=np.array([])
for _ in range(5):
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




k=100

logZ_all1=np.array([])
logZ_all2=np.array([])
for m in range(1):
    logXreal, logL = gen_ns_run(nlive,ndead)
    plt.axvline(logZ(logL,logXreal),color='k')
    logZ_pl=np.ones(1000)
    for _ in range(1000):
        logZ_pl[_]=logZ(logL, logX_gamma_append(logL,nlive,k))
    logZ_all1=np.concatenate((logZ_all1,logZ_pl),axis=None)
    logZ_pl=np.ones(1000)
    for _ in range(1000):
        logZ_pl[_]=logZ(logL, analytical_grad(logL,logXreal,nlive))
    logZ_all2=np.concatenate((logZ_all2,logZ_pl),axis=None)
plt.hist(logZ_all1, alpha=0.5)
error_sum1= np.std(logZ_all1)
print('error is on logZ of gamma append run is',error_sum1)
plt.hist(logZ_all2, color='orange' ,alpha=0.5)
error_sum2= np.std(logZ_all2)
print('error is on logZ of power run is',error_sum2)
plt.subplots()

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
    for k in [ 500,1000,2000,4000]:
        logZ_pl=np.ones(1000)
        for _ in range(1000):
            logZ_pl[_]=logZ(logL[(k//(2))-1:ndead-(k//2)-2], logX_gamma(logL,nlive,k))
        logZ_all3=np.concatenate((logZ_all3,np.mean(logZ_pl)),axis=None)
        stds= np.concatenate((stds,np.std(logZ_pl)),axis=None)
        kval=np.concatenate((kval,k),axis=None)
    plt.errorbar(kval,logZ_all3,yerr=stds,color=colors[m])
    print('k values',kval,'logZ vals',logZ_all3)
print('error is on logZ of gamma2 run is',error_sum,'using k value as',k)
plt.subplots()


logZ_all3=np.array([])
logZ_allreal=np.array([])
for _ in range(1000):
    logXreal, logL = gen_ns_run(nlive,ndead)
    plt.axhline(logZ(logL,logXreal),color='k')
    logZ_pl=logZ(logL[(k//(2))-1:ndead-(k//2)-2], logX_gamma(logL,nlive,k))
    logZactual=logZ(logL,logXreal)
    logZ_all3=np.concatenate((logZ_all3,logZ_pl),axis=None)
    logZ_allreal=np.concatenate((logZ_allreal,logZactual),axis=None)
logZ_error= (logZ_all3-logZ_allreal)**2
error_sum= ((np.sum(logZ_error))**0.5)/(len(logZ_error))
kmatrix=np.concatenate((kmatrix,(np.sum(logZ_all3))/len(logZ_all3)),axis=None)



logZ_all3=np.array([])
logZ_allreal=np.array([])
for _ in range(20000):
    logXreal, logL = gen_ns_run(nlive,ndead)
    plt.axhline(logZ(logL,logXreal),color='k')
    logZ_pl=logZ(logL, logX_powerlaw(logL))
    logZ_all3=np.concatenate((logZ_all3,logZ_pl),axis=None)
error_mat=np.array([np.std(logZ_all3)])
kmatrix=np.array([np.mean(logZ_all3)])

for k in  [4,  6,  8, 10, 12, 14, 16, 18]:
    logZ_all3=np.array([])
    logZ_allreal=np.array([])
    for _ in range(20000):
        logXreal, logL = gen_ns_run(nlive,ndead)
        plt.axhline(logZ(logL,logXreal),color='k')
        logZ_pl=logZ(logL[(k//(2))-1:ndead-(k//2)-2], logX_gamma(logL,nlive,k))
        logZactual=logZ(logL,logXreal)
        logZ_all3=np.concatenate((logZ_all3,logZ_pl),axis=None)
        logZ_allreal=np.concatenate((logZ_allreal,logZactual),axis=None)
    logZ_error= (logZ_all3-logZ_allreal)**2
    error_sum= ((np.sum(logZ_error))**0.5)/(len(logZ_error))
    kmatrix=np.concatenate((kmatrix,(np.sum(logZ_all3))/len(logZ_all3)),axis=None)
    error_mat= np.concatenate((error_mat,np.std(logZ_all3)),axis=None)
    print('error is on logZ of gamma run is',error_sum)
    print('average logZ was',(np.sum(logZ_all3))/len(logZ_all3),'for k value',k)
plt.errorbar([ 1, 4,  6,  8, 10, 12, 14, 16],kmatrix,yerr=error_mat)
plt.axhline(logZ(logL,logXreal),color='k')
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
