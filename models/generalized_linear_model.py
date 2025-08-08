# statisticapy/models/generalized_linear_model.py

import numpy as np
from ..core import BaseModel
from ..utils.formula_parser import parse_formula

# Families (Gaussian, Binomial, Poisson) and other classes remain the same (not repeated here)

class GeneralizedLinearModel(BaseModel):
    """
    Generalized Linear Model with IRLS fitting.
    
    Parameters
    ----------
    family : instance of Family class, default Gaussian
        The family object defining the distribution and link function.
    fit_intercept : bool, default True
        Whether to include an intercept term. Ignored if formula is given.
    formula : str, optional
        Model formula string, e.g. 'y ~ x1 + x2'.
        If provided, fit expects `data` parameter.
    max_iter : int, default 100
        Maximum number of IRLS iterations.
    tol : float, default 1e-6
        Convergence tolerance.
    
    Attributes
    ----------
    params_ : ndarray
        Estimated coefficients.
    fitted : bool
        Indicates if the model has been fit.
    feature_names_ : list of str
        Names of features in design matrix after parsing formula.
    """
    def __init__(self, family=None, fit_intercept=True, formula=None, max_iter=100, tol=1e-6):
        super().__init__()
        self.family = family if family is not None else Gaussian()
        self.fit_intercept = fit_intercept
        self.formula = formula
        self.max_iter = max_iter
        self.tol = tol
        self.feature_names_ = None
    
    
    def fit(self, X=None, y=None, data=None):
        """
        Fit the GLM model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), optional
            Feature matrix.
        y : array-like of shape (n_samples,), optional
            Response vector.
        data : DataFrame or dict-like, optional
            Dataset used with formula if `formula` is provided.
        
        Returns
        -------
        self : object
        """
        if self.formula is not None:
            if data is None:
                raise ValueError("data must be provided when formula is used")
            y, X = parse_formula(self.formula, data)
            # Intercept handled by formula parser automatically
        else:
            if X is None or y is None:
                raise ValueError("X and y must be provided if formula is not used")
            X = np.asarray(X)
            y = np.asarray(y)
            if self.fit_intercept:
                X = np.column_stack((np.ones(X.shape[0]), X))
        
        n_samples, n_features = X.shape
        
        beta = np.zeros(n_features)
        
        for iteration in range(self.max_iter):
            eta = X @ beta
            mu = self.family.link_inv(eta)
            var = self.family.variance(mu)
            var = np.clip(var, 1e-10, None)
            
            z = eta + (y - mu) / (var * self._deriv_link_inv(eta))
            W = 1 / (var * (self._deriv_link_inv(eta) ** 2))
            
            WX = X * W[:, np.newaxis]
            beta_new = np.linalg.pinv(WX.T @ X) @ WX.T @ z
            
            if np.linalg.norm(beta_new - beta) < self.tol:
                beta = beta_new
                break
            
            beta = beta_new
        
        self.params_ = beta
        self.fitted = True
        
        # Store feature names if formula used
        if self.formula is not None and hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f"x{i}" for i in range(X.shape[1])]
        
        return self
    
    def _deriv_link_inv(self, eta):
        # Same as before
    
    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("You must fit the model before prediction.")
        
        X = np.asarray(X)
        if self.formula is None and self.fit_intercept:
            X = np.column_stack((np.ones(X.shape[0]), X))
        elif self.formula is not None:
            # For now, prediction with formula requires DataFrame and parsing - not implemented
            raise NotImplementedError("Prediction with formula requires data frame and parsing is not implemented.")
        
        eta = X @ self.params_
        return self.family.link_inv(eta)

class LogisticRegression(GeneralizedLinearModel):
    def __init__(self, fit_intercept=True, formula=None, max_iter=100, tol=1e-6):
        super().__init__(family=Binomial(), fit_intercept=fit_intercept,
                         formula=formula, max_iter=max_iter, tol=tol)

class PoissonRegression(GeneralizedLinearModel):
    def __init__(self, fit_intercept=True, formula=None, max_iter=100, tol=1e-6):
        super().__init__(family=Poisson(), fit_intercept=fit_intercept,
                         formula=formula, max_iter=max_iter, tol=tol)
