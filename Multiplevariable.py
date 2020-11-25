import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
import seaborn as sns
import arviz as az
from sklearn.preprocessing import scale

df = sns.load_dataset('iris')
print(df.head())
iris = df.query("species == ('setosa', 'versicolor')")
y = pd.Categorical(iris['species']).codes
x = iris[iris.columns[:-1]].values

with pm.Model() as Model:
    alpha = pm.Normal('alpha', mu=0, sigma=100)
    beta = pm.Normal('beta', mu=0, sigma=2, shape=(2))
    mu = alpha + pm.math.dot(x[:, 0:2], beta)
    p = pm.Deterministic('p', pm.math.sigmoid(mu))
    db = pm.Deterministic('db', -(alpha/beta[1]) - (beta[0]/beta[1])*x[:, 0])
    pm.Bernoulli('p-lik', p=p, observed=y)
    trace_m = pm.sample(2000, cores=1)
    #pp = pm.sample_posterior_predictive(trace_m)

_, ax = plt.subplots(figsize=(12, 8))
theta = trace_m['db'].mean(axis=0)
ax.scatter(x[:, 0], x[:, 1], c=[f'C{k}' for k in y])
ix = np.argsort(x[:, 0])
ax.plot(x[:, 0][ix], theta[ix])
az.plot_hdi(x[:, 0], trace_m['db'], color='k', ax=ax)
plt.show()
