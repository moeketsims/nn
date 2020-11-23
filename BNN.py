import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.preprocessing import scale
from scipy.stats import distributions as dist
import arviz as az
import theano
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
# Import the dataset
df = pd.read_csv('heart.csv')
df = df.iloc[0:225, :]

#df = shuffle(df)
# set target variable
y = df['target']
# set st input
X = df.drop(columns='target')
X = scale(X)
n_hidden_units = 5

initial_w1 = np.random.normal(loc=0, scale=10, size=(X.shape[1], n_hidden_units)).astype(theano.config.floatX)
initial_w2 = np.random.normal(loc=0, scale=10, size=(n_hidden_units, n_hidden_units)).astype(theano.config.floatX)
initial_out = np.random.normal(loc=0, scale=10, size=n_hidden_units).astype(theano.config.floatX)

with pm.Model() as BNN:
    x = pm.Data('x', X)
    Y = pm.Data('Y', y)

    weight_1 = pm.Normal('layer_1', mu=0, sd=10, shape=(X.shape[1], n_hidden_units), testval=initial_w1)
    weight_2 = pm.Normal('layer_2', mu=0, sd=10, shape=(n_hidden_units, n_hidden_units), testval=initial_w2)
    weight_Out = pm.Normal('layer_out', mu=0, sd=10, shape=(n_hidden_units,), testval=initial_out)

    layer_1 = pm.math.tanh(pm.math.dot(X, weight_1))
    layer_2 = pm.math.tanh(pm.math.dot(layer_1, weight_2))
    output = pm.math.sigmoid(pm.math.dot(layer_2, weight_Out))

    y_lik = pm.Bernoulli('y_lik', output, observed=Y)
    inference = pm.ADVI()
    approx = pm.fit(3000, method=inference)
    trace = approx.sample(draws=500)
    ppc = pm.sample_posterior_predictive(trace)

pred = ppc['y_lik'].mean(axis=0) > 0
pred = np.round(pred)
print(f'Accuracy is {(pred == y).mean() * 100}')

grid = np.mgrid[-1:1:15j, -1:1:15j]
grid_2d = grid.reshape(2, -1).T
dummy_out = np.ones(grid.shape[1], dtype=np.int8)

with BNN:
    pm.set_data({'x': grid_2d})
    pm.set_data({'Y': dummy_out})
    ppc = pm.sample_ppc(trace, samples=5000)

pred = ppc['y_lik'].mean(axis=0)

# sns.heatmap(pred.reshape(15, 15).T)
#sns.heatmap(ppc['y_lik'].std(axis=0).reshape(15, 15).T)
plt.show()
