from scipy.stats import powerlaw
import numpy as np
import matplotlib.pyplot as plt

# 'PolyChord' parameters
nlive = 500
nprior = nlive*10
ndead = nlive*15

# Generate a dynamic set of live points

# PolyChord cosmo run
nbirth = [nprior] + [-1] * nlive*9 + [0] * ndead + [-1] * (nlive-1)

# Simple run with constant nlive
#nbirth = [nlive] + [0] * (ndead-1)
nlive = np.cumsum(nbirth)


# compute the average logX for the dynamic case
logX_mean = -(1/nlive).cumsum()


# Plot the variable nlive
fig, ax = plt.subplots(3,sharex=True)
ax[0].plot(logX_mean, nlive)
ax[0].set_ylabel(r'$n_\mathrm{live}$')


# Plot the volume differences
for _ in range(10):
     t = powerlaw(nlive).rvs()
     logX = np.log(t).cumsum() - logX_mean
     ax[1].plot(logX_mean, np.log(t).cumsum() - logX_mean ,'C0')

ax[1].set_xlabel(r'$\langle\log X\rangle$')
ax[1].set_ylabel(r'$\log X-\langle\log X\rangle$')


C = 10
sigma = 0.01
def loglikelihood(X):
      return -X**(2/C)/(2*sigma**2)


def gen_ns_run(nlive):
     t = powerlaw(nlive).rvs()
     logX = np.log(t).cumsum()
     logL = loglikelihood(np.exp(logX))
     return logX, logL

_, logL = gen_ns_run(nlive)


y = nlive[1:] * (logL[1:]-logL[:-1])
cumsumy = np.cumsum(y)

k = 50
rollingsumy = cumsumy[k:] - cumsumy[:-k]
theta = (logL[k+1:] - logL[:-k-1]) / rollingsumy
ax[2].plot(logX_mean[:-k-1],theta)
ax[2].set_ylim(0,5/500)
ax[2].set_ylabel(r'$\theta$')

