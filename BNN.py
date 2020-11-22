import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import distributions as dist
import arviz as az
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# Import the dataset
df = pd.read_csv('heart.csv')
# set target variable
y = df['target']
print(y.unique())
# set st input
X = df.drop(columns='target')
n_hidden_units = 5

initial_w1 = np.random.normal(loc=0, scale=100, size=(X.shape[1], n_hidden_units))
initial_w2 = np.random.normal(loc=0, scale=100, size=(n_hidden_units, n_hidden_units))
initial_out = np.random.normal(loc=0, scale=100, size=n_hidden_units)

with pm.Model() as BNN:
    x = pm.Data('x', X)
    y = pm.Data('y', y)
    weight_1 = pm.Normal('layer_1', mu=10, sd=1, shape=(X.shape[1], n_hidden_units), testval=initial_w1)
    weight_2 = pm.Normal('layer_2', mu=10, sd=1, shape=(n_hidden_units, n_hidden_units), testval=initial_w2)
    weight_Out = pm.Normal('layer_out', mu=10, sd=1, shape=(n_hidden_units,), testval=initial_out)

    layer_1 = pm.math.tanh(pm.math.dot(X, weight_1))
    layer_2 = pm.math.tanh(pm.math.dot(layer_1, weight_2))
    output = pm.math.sigmoid(pm.math.tanh(pm.math.dot(layer_2, weight_Out)))
    y_lik = pm.Bernoulli('y_lik', output, observed=y)
    inference = pm.ADVI()
    approx = pm.fit(n=5000, method=inference)
    trace = approx.sample(draws=5000)
    ppc = pm.sample_posterior_predictive(trace, samples=500)
    #trace_m = pm.sample(5000, tune=3000, random_seed=123, cores=1)
    #ppc = pm.sample_posterior_predictive(trace_m, random_seed=123)

pred = ppc['y_lik'].mean(axis=0) > 0.5

grid = np.mgrid[-3:3:100j, -3:3:100j]
grid_2d = grid.reshape(2, -1).T
dummy_out = np.ones(grid.shape[1], dtype=np.int8)

with BNN:
    pm.set_data({'x': grid_2d})
    pm.set_data({'y': dummy_out})
    ppc = pm.sample_ppc(trace, samples=500)


pred = ppc['y_lik'].mean(axis=0)
print(pred)
plt.plot(pred)
#sns.heatmap(pred.reshape(102, 102).T)
#plt.show()
#sns.heatmap(ppc['y_lik'].std(axis=0).reshape(303, 303).T)

print(X.head())