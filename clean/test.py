from nestedsampling.samplers import OrthodoxNS, MetropolisNS
import numpy as np
from scipy.stats import multivariate_normal

def loglikelihood(x):
    mean = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    cov = np.array([[0.01, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.01, 0.0, 0.0, 0.0, 0.0 ], [0.0, 0.0, 0.01, 0.0, 0.0, 0.0],[ 0.0, 0.0, 0.0, 0.01,0.0,0.0],[0.0, 0.0, 0.0, 0.0, 0.01,0.0],[0.0, 0.0, 0.0, 0.0, 0.0,0.01]])
    logL = multivariate_normal.logpdf(x, mean=mean, cov=cov) + np.random.rand()*sigma_
    return logL

sigma_ = 0.0
mh= MetropolisNS(loglikelihood=loglikelihood,prior_bounds=[[0,0,0,0,0,0],[1,1,1,1,1,1]],ndims=6,sigma=sigma_)

fig,axs=mh.MCMC2d(parameters=['x0','x1'],label_='deterministic')

for sigma_ in [0.25,20.0,100.0]:    
    mh= MetropolisNS(loglikelihood=loglikelihood,prior_bounds=[[0,0,0,0,0,0],[1,1,1,1,1,1]],ndims=6,sigma=sigma_)
    mh.MCMC2d(parameters=axs,label_='non-deterministic with std error='+str(sigma_))

    
handles, labels = axs['x0']['x1'].get_legend_handles_labels()
leg = fig.legend(handles, labels)
fig.tight_layout()
plt.show()

