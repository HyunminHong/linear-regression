from src.OLS import OLS
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import reshape
import statsmodels.api as sm
from sklearn import datasets

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Train the model using the training sets
model1 = OLS()
fitted_mod1 = model1.fit(diabetes_X_train, diabetes_y_train)
fitted_mod1.beta
fitted_mod1.rank
fitted_mod1._dof_model
fitted_mod1._dof_resid
diabetes_y_pred = fitted_mod1.predict(diabetes_X_test)
fitted_mod1.rsquared()
fitted_mod1.rsquared_adj()

plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)
plt.show()

"""
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

plt.scatter(X[:,2], y)
plt.show()
"""