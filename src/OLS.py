import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import reshape

import statsmodels.api as sm

class OLS:
    """
    X: an exogenous variable is one whose value is determined outside the model and is imposed on the model
    y: an endogenous variable is a variable whose value is determined by the model
    """
    def __init__(self, intercept = True):
        self.intercept = intercept # in default, the liear model contains an intercept term (as in practice)
        self.rank = None # rank of the design matrix X
        self._dof_model = None # model degrees of freedom 
        self._dof_resid = None # residual degrees of freedom
        self.beta = None # regression coefficients
        self.X = None
        self.y = None
        # TODO if X is a vector, then X.shape[1] does not exist. 
        self.nob = None # number of observations
        self.dim = None # dimension of the data (in Hayashi, dim refers to K) 
        self.y_pred = None # predicted value of y based on OLS estimate

    def rank_exog(self):
        """
        return the rank of the exogenous matrix
        """
        if self.rank == None:
            self.rank = np.linalg.matrix_rank(self.X)
        return self.rank

    # TODO take care of a case when there is no intercept
    def dof_model(self):
        """
        model degrees of freedom is defined by:

        (rank of X) - 1
        """
        self._dof_model = self.rank_exog() - 1
        return self._dof_model

    def set_dof_model(self, value):
        """
        setter function for the model degrees of freedom 
        """
        self._dof_model = value
        return self._dof_model
    
    # TODO take care of a case when there is no intercept
    def dof_resid(self):
        """
        residual degrees of freedom is defined by:

        # observations - (rank of X)
        """
        self._dof_resid = self.nob - self.rank_exog()
        return self._dof_resid

    def set_dof_resid(self, value):
        """
        setter function for the residual degrees of freedom 
        """
        self._dof_resid = value
        return self._dof_resid

    def fit(self, X, y, method = "qr"):
        """
        Through the QR-decomposition of the X matrix, we can compute the least-squares coefficients. 
        X = Q * R where Q is an orthogonal matrix and R is an upper triangular matrix.

        We solve for beta: 
        X.T * X * beta = X.T * y

        Then, the LHS can be written as: 
        R.T * (Q.T * Q) * R = R.T * R due to the orthogonality. 
        
        Hence, we then have:
        R.T * R * beta = R.T * Q.T * y => R * beta = Q.T * y
        """
        self.X = X
        self.y = y
        self.nob = self.X.shape[0] # number of observations
        self.dim = self.X.shape[1] # dimension of the data

        try: # X.T * X is a matrix
            if method == "qr":
                Q, R = np.linalg.qr(self.X)
                self.beta = np.linalg.solve(R, np.dot(Q.T, self.y))

            elif method == "conv":
                """
                conventional way of computing beta:

                beta = (X.T * X)^(-1) * X.T * y
                """
                self.beta = np.linalg.solve(np.dot(self.X.T, self.X), np.dot(self.X.T, self.y))
            return self

        except np.linalg.LinAlgError: # X.T * X is a constant, i.e., X is a nx1 vector
            self.beta = np.divide(np.dot(self.X.T, self.y), np.dot(self.X.T, self.X))
            return self

    def predict(self, X_test):
        """
        y_pred = _X*beta where beta is the OLS estimate

        example.
        y = a + b*X1 + c*X2 = X * beta where X = [1 X2 X3] and beta = [a b c].T

        y_pred = X_test * beta
        """
        self.y_pred = np.dot(X_test, self.beta)
        return self.y_pred

    def rss(self):
        """
        residual sum of errors (RSS). (it is equivalent to SSR in Hayashi)

        resid.T * resid
        """
        resid = self.y - np.dot(self.X, self.beta)
        self.rss = np.dot(resid.T, resid)
        return self.rss

    def tss(self):
        """
        total sum of squares (TSS).

        (y - mean(y)).T * (y - mean(y))

        if it has no intercept, no need to center, i.e,. y.T * y
        """
        if self.intercept:
            y_centered = self.y - np.mean(self.y)
            self.tss = np.dot(y_centered.T, y_centered)
            return self.tss
            
        else:
            self.tss = np.dot(self.y, self.y)
            return self.tss

    def ess(self):
        """
        explained sum of squares (ESS).

        (y_pred - mean(y)).T * (y_pred - mean(y))

        if it has no intercept, no need to center, i.e,. y_pred.T * y_pred
        """
        self.ess = self.tss - self.rss
        return self.ess

    def r_squared(self):
        """
        Note that: 
        * TSS = ESS + RSS 
        * Rsquared = 1 - RSS/TSS
        """
        # TODO use attribute instead of method
        return 1 - self.rss()/self.tss()

    def r_squared_adj(self):
        """
        adjusted Rsquared = 1 - (1 - Rsquared)*(N - 1)/(N - p - 1)

        if no intercept is given, then no -1 term in denominator
        """
        return 1 - self.r_squared()

    def visualize(self):
        return 1



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

lm = OLS()
lm.fit(X, y, method = "qr").predict(_X)
lm.fit(X, y).r_squared()
lm.r_squared()


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
