# statisticapy/models/linear_model.py

import numpy as np
from ..core import BaseModel

class LinearRegression(BaseModel):
    """
    Ordinary Least Squares (OLS) Linear Regression
    
    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    
    Attributes
    ----------
    params_ : ndarray of shape (n_features,) or (n_features + 1,)
        Estimated coefficients including intercept if fit_intercept=True.
    fitted : bool
        Whether the model has been fitted.
    
    Examples
    --------
    >>> import numpy as np
    >>> from statisticapy.models.linear_model import LinearRegression
    >>> X = np.array([[1, 2], [2, 3], [4, 5], [3, 2]])
    >>> y = np.array([3, 5, 9, 6])
    >>> model = LinearRegression()
    >>> model.fit(X, y)
    >>> preds = model.predict(X)
    >>> print(preds)
    """
    def __init__(self, fit_intercept=True):
        super().__init__()
        self.fit_intercept = fit_intercept
    
    def fit(self, X, y):
        """
        Fit the linear model using Ordinary Least Squares.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if self.fit_intercept:
            # Add intercept column of ones
            X = np.column_stack((np.ones(X.shape[0]), X))
        
        # Normal equation: params = (X'X)^-1 X'y
        XtX = X.T @ X
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            raise ValueError("Singular matrix detected during fit; possibly multicollinearity.")
        
        Xty = X.T @ y
        self.params_ = XtX_inv @ Xty
        self.fitted = True
    
    def predict(self, X):
        """
        Predict using the linear model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values
        """
        if not self.fitted:
            raise RuntimeError("You must fit the model before prediction.")
        
        X = np.asarray(X)
        if self.fit_intercept:
            X = np.column_stack((np.ones(X.shape[0]), X))
        
        return X @ self.params_
