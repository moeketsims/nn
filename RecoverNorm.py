import pymc3 as pm
import arviz as az
from scipy import stats as st
import matplotlib.pyplot as plt

n=1000
xs = st.bernoulli(0.8).rvs(n)
with pm.Model() as RecoverNorm:
    theta = pm.Beta('mu', 5, 1)
    pm.Bernoulli('x', p=theta, observed=xs)
    trace = pm.sample(draws=2000, cores=1, random_seed=123, chains=4)
az.plot_trace(trace)
plt.show()
print(az.summary(trace))
