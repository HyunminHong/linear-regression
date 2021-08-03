import numpy as np

class LinearModel:
    """
    base class for linear regression models such as OLS
    """
    def __init__(self, X = None, y = None, intercept = True):
        self.X = X
        self.y = y
        
    def rank_exog(self):
        """
        return the rank of the exogenous matrix
        """
        if self.rank == None:
            self.rank = np.linalg.matrix_rank(self.X)
        return self.rank

    def dof_model(self):
        """
        model degrees of freedom is defined by:

        (rank of X) - 1
        """
        self._dof_model = self.rank_exog() - self.intercept
        return self._dof_model

    def set_dof_model(self, value):
        """
        setter function for the model degrees of freedom 
        """
        self._dof_model = value
        return self._dof_model
    
    def dof_resid(self):
        """
        residual degrees of freedom is defined by:

        # observations - (rank of X)
        """
        self._dof_resid = self.nob - self.rank_exog() - self.intercept
        return self._dof_resid

    def set_dof_resid(self, value):
        """
        setter function for the residual degrees of freedom 
        """
        self._dof_resid = value
        return self._dof_resid

