import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import distributions as dist
import theano
import arviz as az
import warnings

warnings.filterwarnings('ignore')

x = np.array([1, 5, 8])
y = 1 + x + dist.norm(0, 1.5).rvs(3)
y = y.reshape(-1, 1)
x = x.reshape(-1, 1)
xs = x
ys = y
print(x.shape)

initial_w = dist.uniform(0, 1).rvs(1)

with pm.Model() as NN:
    x = pm.Data('x', x)
    y = pm.Data('y', y)
    w = pm.Normal('w', mu=0, sigma=10, shape=[1, 1], testval=initial_w)
    l1 = pm.Deterministic('l1', pm.math.dot(x, w))
    y_l = pm.Normal('y_l', l1, observed=y)
    trace = pm.sample(cores=1)
    pp = pm.sample_posterior_predictive(trace, random_seed=123)


plt.plot(xs, pp['y_l'].mean(axis=0).reshape(1, -1).flatten())
plt.scatter(xs, ys)
plt.show()
