from nestedsamplingbreak.samplers import NSrun
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import sys
import pandas as pd

def loglikelihood(x):
    mean = np.array([0.5, 0.5])
    cov = np.array([[0.01, 0.0], [0.0, 0.01 ]])
    logL = multivariate_normal.logpdf(x, mean=mean, cov=cov) + np.random.normal(0,sigma_)
    prob = np.random.rand()
    if prob>0.99999: 
        print('loglikelihood is using sigma_ as'+str(sigma_))
    return logL


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))/(sig*np.sqrt(2*np.pi))


sigma_ = 0
mh1= NSrun(loglikelihood=loglikelihood,prior_bounds=[[0,0],[1,1]],ndims=2,tol=1e-7,multi_samples=1)

fig,axs=mh1.MCMC2d(parameters=['x0','x1'],label_='deterministic orthodox')


# for sigma_ in [2.0]:    
#     mh_err= NSrun(loglikelihood=loglikelihood,prior_bounds=[[0,0], [1,1]], ndims=2,tol=1e-35,multi_samples=100)
#     mh_err.MCMC2d(parameters=axs,label_='Multi Sampler with std error='+str(sigma_))


# for sigma_ in [.20]:    
#     mh_err= NSrun(loglikelihood=loglikelihood,prior_bounds=[[0,0], [1,1]], ndims=2,tol=1e-35,multi_samples=1)
#     mh_err.MCMC2d(parameters=axs,label_='Non-Multi Sampler with std error='+str(sigma_))


# handles, labels = axs['x0']['x0'].get_legend_handles_labels()
# leg = fig.legend(handles, labels)
# fig.tight_layout()
# plt.show()


# plt.figure(0)
# x = np.linspace(0,1,3000)
  
# mean = 0.5, 
# sigg = 0.1
# y = gaussian(x, mu=mean, sig=sigg)



# axs['x0']['x0'].plot(x,y,label= 'gaussian')

# axs['x1']['x1'].plot(x,y,label= 'gaussian')


# handles, labels = axs['x0']['x1'].get_legend_handles_labels()
# leg = fig.legend(handles, labels)
# fig.tight_layout()
# plt.show()

# sys.exit(0)

#####################################################
'''%Metropolis hastings below'''


# sigma_ = 0.0
# mh= NSrun(loglikelihood=loglikelihood,prior_bounds=[[0,0],[1,1]],ndims=2,Metropolis=True,multi_samples=1)

# fig,axs=mh.MCMC2d(parameters=['x0','x1'],label_='deterministic Metropolis')

# sigma_=2.0

for sigma_ in [0]:    
    mh2= NSrun(loglikelihood=loglikelihood,prior_bounds=[[0,0], [1,1]], ndims=2,Metropolis=True,multi_samples=1,tol=1e-9,num_repeats= 100)
    mh2.MCMC2d(parameters=axs,label_='non-deterministic Metropolis with std error='+str(sigma_))
    
# sigma_=20    

# for sigma_ in [20.0]:    
#     mh3= NSrun(loglikelihood=loglikelihood,prior_bounds=[[0,0], [1,1]], ndims=2,Metropolis=True,multi_samples=100)
#     mh3.MCMC2d(parameters=axs,label_='non-deterministic Metropolis with std error='+str(sigma_))



# with std error='+str(sigma_))


# x = np.linspace(0,1,3000)

# mean = 0.5, 
# sigg = 0.1
# y = gaussian(x, mu=mean, sig=sigg)

# axs['x0']['x0'].plot(x,y,color='purple')
    


# axs['x1']['x1'].plot(x,y,color='purple')
    
#  console 7 is running A=1 and 50 num steps and 100 multi samples
# console 8 is running A=1 and 4200 step number with 1 multi samples
# console 6 is running 100 step number and A=300 and 100 multi samples


handles, labels = axs['x0']['x0'].get_legend_handles_labels()
leg = fig.legend(handles, labels)
fig.tight_layout()
plt.show()


plt.figure(0)

# plt.hist(mh1.logZ_int(200),label= 'ORTHODOX', alpha=0.5)
# plt.hist(mh2.logZ_int(200),label= 'MCMC',alpha=0.5)

# plt.legend(loc="upper left")








# # df = pd.DataFrame({
# #     'x_vals':x_
# # })
# # df['Actual Gaussian']=y

# # df.plot(x=x_,y='Actual Gaussian',ax=axs['x0']['x0'],color='purple')

# # df.plot(x=x_,y='Actual Gaussian',ax=axs['x1']['x1'],color='purple')
