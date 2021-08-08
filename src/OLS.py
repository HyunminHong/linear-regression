from src.LinearModel import LinearModel
import numpy as np
import matplotlib.pyplot as plt

class OLS(LinearModel):
    """
    X: an exogenous variable is one whose value is determined outside the model and is imposed on the model
    y: an endogenous variable is a variable whose value is determined by the model
    """
    def __init__(self, X = None, y = None, intercept = True):
        super().__init__(X = None, y = None, intercept = True)
        self.rank = None # rank of the design matrix X
        self._dof_model = None # model degrees of freedom 
        self._dof_resid = None # residual degrees of freedom
        self.beta = None # regression coefficients
        # TODO if X is a vector, then X.shape[1] does not exist. 
        self.nob = None # number of observations
        self.y_pred = None # predicted value of y based on OLS estimate
        self.r_squared = None
        self.r_squared_adj = None

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
        if self.X == None:
            self.X = X

        if self.y == None:
            self.y = y
        
        self.nob = X.shape[0]

        self.rank_exog()
        self.dof_model()
        self.dof_resid()

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

    def rss_calc(self):
        """
        residual sum of errors (RSS). (it is equivalent to SSR in Hayashi)

        resid.T * resid
        """
        resid = self.y - np.dot(self.X, self.beta)
        self.rss = np.dot(resid, resid)

    def tss_calc(self):
        """
        total sum of squares (TSS).

        (y - mean(y)).T * (y - mean(y))

        if it has no intercept, no need to center, i.e,. y.T * y
        """
        if self.intercept:
            y_centered = self.y - np.mean(self.y)
            self.tss = np.dot(y_centered, y_centered)
            
        else:
            self.tss = np.dot(self.y, self.y)

    def ess_calc(self):
        """
        explained sum of squares (ESS).

        (y_pred - mean(y)).T * (y_pred - mean(y))

        if it has no intercept, no need to center, i.e,. y_pred.T * y_pred
        """
        self.rss_calc()
        self.tss_calc()
        self.ess = self.tss - self.rss

    def rsquared(self):
        """
        Note that: 
        * TSS = ESS + RSS 
        * Rsquared = 1 - RSS/TSS
        """
        self.rss_calc()
        self.tss_calc()
        if self.r_squared == None:
            self.r_squared = 1 - np.divide(self.rss, self.tss)
        return self.r_squared

    #TODO
    def rsquared_adj(self):
        """
        adjusted Rsquared = 1 - (1 - Rsquared)*(N - 1)/(N - p - 1)

        if no intercept is given, then no -1 term in denominator
        """
        self.rss_calc()
        self.tss_calc()
        self.rsquared()
        return 1 - ((1 - self.r_squared) * np.divide(self.nob - self.intercept, self._dof_resid))

    def plot_regression(self, method = "PCA"):
        if method == "PCA":
            """
            plot the regression line using PCA (dimension reduction)

            TODO can we do PCA without using Scikit learn? 
            """
            pass

        if method == "proj":
            """
            plot on projection
            """
            pass