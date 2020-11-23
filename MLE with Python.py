
from scipy import stats as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


intercept_true = 5
coef1 = 0.8
coef2 = 3.0
n = 100
x = st.uniform(0, 30).rvs(n)
#y_true = intercept_true + slope_true * x
y_obs = intercept_true + coef1 * x + coef2 * x**2 + st.norm(0, 5).rvs(n)

X = np.c_[np.ones(len(x)), x, x**2]

theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y_obs)

sigma = (1/len(x)) * ((y_obs - X.dot(theta)).T.dot((y_obs - X.dot(theta))))

xs = np.linspace(0, 30, 100)
y_ = st.norm(theta[0] + theta[1] * xs + theta[2] * xs**2, sigma).rvs(len(xs))

_, axiz = plt.subplots(figsize=(12, 8))
sns.scatterplot(x, y_obs, color='k', ax=axiz)
sns.lineplot(x, theta[0] + theta[1] * x + theta[2] * x**2)
sns.scatterplot(xs, y_, color='red', ax=axiz)
plt.legend(['MLE_mean', 'observed', 'Probabilistic'])

plt.show()

print(f'intercept {theta[0]} the theta_1 {theta[1]} theta_2 {theta[2]} ')




