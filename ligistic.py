import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az

x = np.linspace(-10, 10, 10000)
y = 1 / (1 + np.exp(-x))
plt.plot(x, y)
plt.show()
