import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import distributions as dist
import arviz as az
import warnings
warnings.filterwarnings('ignore')

n = 1000 # number points
# simulate from uniform distribution
x = dist.uniform(0, 30).rvs(n)
x = x[np.argsort(x)]

theta_0 = 1.0
theta_1 = 3.2
theta_2 = 4.0

# true model of the world
y_true = theta_0 + theta_1 * x + theta_2 * x**2

# Observed data with noise
y_obs = y_true + dist.norm(0, 100).rvs(n)

# Model description
with pm.Model() as linear_Model:
    xs = pm.Data('xs', x)
    y = pm.Data('y', y_obs)
    theta_0 = pm.Normal('intercept', mu=0, sigma=2)
    theta_1 = pm.Normal('coefx', mu=0, sigma=2)
    theta_2 = pm.Normal('coefxSqd', mu=0, sigma=2)
    theta = pm.Deterministic('theta', theta_0 + theta_1*xs + theta_2*xs**2)
    sigma = pm.HalfCauchy('sigma', 100)
    y_lik = pm.Normal('y_lik', mu=theta, sigma=sigma, observed=y)
    trace_linear = pm.sample(tune=2000, chains=1, cores=1)
    pp_samples = pm.sample_posterior_predictive(trace=trace_linear, random_seed=123)

y_pred = pp_samples['y_lik'].mean(axis=0)

_, axi = plt.subplots(1, 4, figsize=(8, 5))
sns.scatterplot(x, y_obs, ax=axi[0]).set_title("Data")
sns.lineplot(x, y_pred, ax=axi[0])
az.plot_hdi(x, trace_linear['theta'], hdi_prob=0.98, ax=axi[0], color='gray')
az.plot_posterior(trace_linear, var_names=['intercept', 'coefx'], ax=axi[1])
az.plot_posterior(trace_linear, var_names=['coefx'], ax=axi[2])
az.plot_posterior(trace_linear, var_names=['coefxSqd'], ax=axi[3])
plt.show()


with linear_Model:
    pm.set_data({'xs': [1, 5.6, 4]})
    y_test = pm.sample_posterior_predictive(trace=trace_linear)
print(y_test['y_lik'].mean(axis=0))
print(1 + 3.2 * 1 + 4 * 1**2)









