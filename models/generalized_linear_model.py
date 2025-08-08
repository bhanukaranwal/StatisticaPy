# statisticapy/models/generalized_linear_model.py

import numpy as np
from ..core import BaseModel

class Family:
    """
    Base class for GLM family. Defines variance and link functions.
    """
    def variance(self, mu):
        raise NotImplementedError()
    
    def link(self, mu):
        raise NotImplementedError()
    
    def link_inv(self, eta):
        raise NotImplementedError()
    
    def deviance(self, y, mu):
        raise NotImplementedError()

class Gaussian(Family):
    def variance(self, mu):
        return np.ones_like(mu)
    
    def link(self, mu):
        return mu
    
    def link_inv(self, eta):
        return eta
    
    def deviance(self, y, mu):
        return np.sum((y - mu)**2)

class Binomial(Family):
    def variance(self, mu):
        return mu * (1 - mu)
    
    def link(self, mu):
        return np.log(mu / (1 - mu))
    
    def link_inv(self, eta):
        return 1 / (1 + np.exp(-eta))
    
    def deviance(self, y, mu):
        # Binomial deviance for y in {0,1}
        eps = 1e-9
        mu = np.clip(mu, eps, 1 - eps)
        y = np.clip(y, eps, 1 - eps)
        return 2 * np.sum(y * np.log(y / mu) + (1 - y) * np.log((1 - y) / (1 - mu)))

class Poisson(Family):
    def variance(self, mu):
        return mu
    
    def link(self, mu):
        return np.log(mu)
    
    def link_inv(self, eta):
        return np.exp(eta)
    
    def deviance(self, y, mu):
        eps = 1e-9
        mu = np.clip(mu, eps, None)
        y = np.clip(y, eps, None)
        return 2 * np.sum(y * np.log(y / mu) - (y - mu))

class GeneralizedLinearModel(BaseModel):
    """
    Generalized Linear Model with IRLS fitting.
    
    Parameters
    ----------
    family : instance of Family class, default Gaussian
        The family object defining the distribution and link function.
    fit_intercept : bool, default=True
        Whether to include an intercept term.
    max_iter : int, default 100
        Maximum number of IRLS iterations.
    tol : float, default 1e-6
        Convergence tolerance for stopping criterion.
    
    Attributes
    ----------
    params_ : ndarray
        Estimated coefficients.
    """
    def __init__(self, family=None, fit_intercept=True, max_iter=100, tol=1e-6):
        super().__init__()
        self.family = family if family is not None else Gaussian()
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
    
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        
        if self.fit_intercept:
            X = np.column_stack((np.ones(X.shape[0]), X))
        
        n_samples, n_features = X.shape
        
        # Initialize parameters
        beta = np.zeros(n_features)
        
        for iteration in range(self.max_iter):
            eta = X @ beta
            mu = self.family.link_inv(eta)
            var = self.family.variance(mu)
            # Avoid division by zero
            var = np.clip(var, 1e-10, None)
            
            # Calculate weights and working response
            z = eta + (y - mu) / (var * self._deriv_link_inv(eta))
            W = 1 / (var * (self._deriv_link_inv(eta) ** 2))
            
            # Weighted least squares step
            WX = X * W[:, np.newaxis]
            beta_new = np.linalg.pinv(WX.T @ X) @ WX.T @ z
            
            # Check convergence
            if np.linalg.norm(beta_new - beta) < self.tol:
                beta = beta_new
                break
            
            beta = beta_new
        
        self.params_ = beta
        self.fitted = True
    
    def _deriv_link_inv(self, eta):
        """
        Derivative of the inverse link function w.r.t eta.
        Numerical derivative as default; subclasses can override.
        """
        h = 1e-8
        return (self.family.link_inv(eta + h) - self.family.link_inv(eta - h)) / (2 * h)
    
    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("You must fit the model before prediction.")
        
        X = np.asarray(X)
        if self.fit_intercept:
            X = np.column_stack((np.ones(X.shape[0]), X))
        
        eta = X @ self.params_
        return self.family.link_inv(eta)

class LogisticRegression(GeneralizedLinearModel):
    def __init__(self, fit_intercept=True, max_iter=100, tol=1e-6):
        super().__init__(family=Binomial(), fit_intercept=fit_intercept, max_iter=max_iter, tol=tol)

class PoissonRegression(GeneralizedLinearModel):
    def __init__(self, fit_intercept=True, max_iter=100, tol=1e-6):
        super().__init__(family=Poisson(), fit_intercept=fit_intercept, max_iter=max_iter, tol=tol)
