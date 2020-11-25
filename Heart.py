import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
import seaborn as sns
import arviz as az

df = sns.load_dataset('iris')
iris = df.query("species == ('setosa', 'versicolor')")
y = pd.Categorical(iris['species']).codes
x = iris[iris.columns[:-1]].values
x = x[:, 0] - x[:, 0].mean()
print(x)
with pm.Model() as model:
    alpha = pm.Normal('alpha', 0, 10)
    beta = pm.Normal('beta', 0, 10)
    mu = alpha + pm.math.dot(x, beta)
    p = pm.Deterministic('p', pm.math.sigmoid(mu))
    y_lik = pm.Bernoulli('y_lik', p=p, observed=y)
    b = pm.Deterministic('b', -alpha/beta)
    trace_m = pm.sample(draws=1000, cores=1, chains=3, random_seed=1)
    pp = pm.sample_posterior_predictive(trace_m)

_, ax = plt.subplots(figsize=(12, 8))
xs = np.linspace(x.min(), x.max(), 1000)
theta = trace_m['p'].mean(axis=0)
sns.lineplot(xs, 1/(1+np.exp(-(trace_m['alpha'].mean(axis=0)+trace_m['beta'].mean(axis=0)*xs))), ax=ax)
plt.vlines(trace_m['b'].mean(axis=0), 0, 1)
az.plot_hdi(x, trace_m['p'], ax=ax)
hdi = az.hdi(trace_m['b'], hdi_prob=0.98)
plt.fill_betweenx([0, 1], hdi[0], hdi[1], color='k', alpha=0.5)
sns.scatterplot(x, y, ax=ax)
plt.xlabel('sepal_length')
plt.show()


