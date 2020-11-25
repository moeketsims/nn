import pymc3 as pm
import arviz as az
from scipy import stats as st
import matplotlib.pyplot as plt
import numpy as np
import graphviz

n = 50
theta_0 = 2
theta_1 = 0.5
xs = st.uniform(0, 30).rvs(n)
y_true = theta_0 + theta_1 * xs
y_obs = y_true + st.norm(0, 0.5).rvs(n)

with pm.Model() as BRegression:
    theta_0 = pm.Normal('theta_0', mu=0, sigma=10)
    theta_1 = pm.Normal('theta_1', mu=0, sigma=10)
    sigma = pm.HalfCauchy('sigma', 10)
    mu = pm.Deterministic('mu', theta_0 + theta_1*xs)
    pm.Normal('y_lik', mu=mu, sigma=sigma, observed=y_obs)
    model_trace = pm.sample(draws=5000, tune=2000, cores=1, chains=4)
    pp = pm.sample_posterior_predictive(trace=model_trace)

_, axi = plt.subplots(figsize=(12, 5))
axi.plot(xs, pp['y_lik'].mean(axis=0), c='k')
az.plot_hdi(xs, model_trace['mu'], hdi_prob=0.98, ax=axi, color='gray')
axi.scatter(xs, y_obs)
plt.ylabel('y_observed', rotation=0, labelpad=30)
az.plot_posterior(model_trace, var_names=['theta_0', 'theta_1'])
plt.show()
#pm.model_to_graphviz(BRegression).view()









