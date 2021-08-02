from src import OLS
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
_X

X = sm.add_constant(X)
y = np.dot(X, beta) + e

lm = OLS(X, y)
lm.fit(method = "qr").predict(_X)
lm.fit().r_squared_adj()
lm.fit().r_squared()
lm.r_squared()

np.divide(lm.nob - lm.intercept, lm._dof_resid)

1 - ((1 - lm.r_squared()) * np.divide(lm.nob - lm.intercept, lm.dof_resid()))

(1 - lm.r_squared())

lm.r_squared_adj()

np.dot(X, lm.beta).shape

np.dot(_X, lm.beta)

model = sm.OLS(y, X)
model.fit().summary()
model.fit().rsquared()
model.fit().predict(_X)

y_easy = np.array([1, 0, 2])
x_easy = np.array([0, 1, 2]).reshape(-1,1)
lm_easy = OLS(x_easy, y_easy)
lm_easy.fit(method = "conv")
model_easy = sm.OLS(y_easy, x_easy)
model_easy.fit(method="pinv").summary()

np.dot(x_easy.T, x_easy) * beta = np.dot(x_easy.T, y_easy)

beta = np.dot(x_easy.T, y_easy)/np.dot(x_easy.T, x_easy)
beta

X1 = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y1 = np.array([5, 20, 14, 32, 22, 38])
lm1 = OLS(X1, y1)
lm1.fit(method = "qr")
model1 = sm.OLS(y1, X1)
model1.fit(method="qr").summary()
model1.k_constant
