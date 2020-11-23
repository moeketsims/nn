from scipy import stats as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

n = 10000
_, axis = plt.subplots(figsize=(10, 4))

sns.distplot(st.t(20).rvs(n), ax=axis)
sns.distplot(st.t(30).rvs(n), ax=axis)
sns.distplot(st.t(70).rvs(n), ax=axis)
sns.distplot(st.t(100).rvs(n), ax=axis)
plt.legend(['Student-T(20)', 'Student-T(30)', 'Student-T(70)', 'Student-T(100)'])
plt.show()