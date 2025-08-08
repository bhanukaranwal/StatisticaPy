# statisticapy/core.py

import numpy as np

class BaseModel:
    """
    Base class for all statistical models in StatisticaPy.
    Provides fit and predict interface.
    """
    def __init__(self):
        self.params_ = None
        self.fitted = False
    
    def fit(self, X, y):
        raise NotImplementedError("Fit method must be implemented by subclasses.")
    
    def predict(self, X):
        raise NotImplementedError("Predict method must be implemented by subclasses.")

def fit_model(model_class, X, y, **kwargs):
    """
    Convenience function to instantiate and fit a model.
    
    Parameters:
        model_class: subclass of BaseModel
        X: array-like, feature matrix
        y: array-like, target vector
        kwargs: additional model-specific keyword arguments
    
    Returns:
        fitted model instance
    """
    model = model_class(**kwargs)
    model.fit(X, y)
    return model
