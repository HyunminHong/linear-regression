from src.OLS import OLS
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import reshape
import statsmodels.api as sm

nsample = 100
x = np.linspace(0, 10, 100)
X = np.column_stack((x, x**2))
beta = np.array([1, 0.1, 10])
e = np.random.normal(size=nsample)

# y_pred = X*beta 
# beta = 3x1
# _X = 1x3

_X = np.array([[1, 1, 1], [2, 2, 2]])
X = sm.add_constant(X)
y = np.dot(X, beta) + e

model1 = OLS()
fitted_mod1 = model1.fit(X, y)
fitted_mod1.beta
fitted_mod1.X
fitted_mod1.y
fitted_mod1.predict(_X)
fitted_mod1.rss_calc()
fitted_mod1.rss
fitted_mod1.tss_calc()
fitted_mod1.tss 
fitted_mod1.ess_calc()
fitted_mod1.ess
fitted_mod1.r_squared()

# OLS from statsmodels.regression.linear_model
model2 = sm.OLS(y, X)
fitted_mod2 = model2.fit()
fitted_mod2.predict(_X)
fitted_mod2.summary()
