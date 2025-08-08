# tests/test_generalized_linear_model.py

import numpy as np
import pytest
from statisticapy.models.generalized_linear_model import LogisticRegression, PoissonRegression

def test_logistic_regression_fit_predict():
    X = np.array([[0.5, 1.2],
                  [1.3, 3.1],
                  [1.8, 2.8],
                  [3.0, 0.5],
                  [2.9, 2.7]])
    y = np.array([0, 1, 1, 0, 1])
    
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    preds = model.predict(X)
    
    # Predicted probabilities should be between 0 and 1
    assert np.all(preds >= 0) and np.all(preds <= 1)
    
    # Predictions roughly correspond to y (check mean approx.)
    assert abs(preds.mean() - y.mean()) < 0.3

def test_poisson_regression_fit_predict():
    X = np.array([[1.0, 0.5],
                  [2.0, 1.5],
                  [3.0, 3.5],
                  [4.0, 0.5],
                  [5.0, 2.0]])
    y = np.array([1, 3, 7, 6, 8])
    
    model = PoissonRegression(max_iter=200)
    model.fit(X, y)
    preds = model.predict(X)
    
    # Predicted means should be non-negative
    assert np.all(preds >= 0)
    
    # Predictions roughly correspond to y mean
    assert abs(preds.mean() - y.mean()) < 2.0

def test_predict_before_fit_raises():
    model = LogisticRegression()
    X = np.array([[1, 2]])
    with pytest.raises(RuntimeError):
        model.predict(X)

if __name__ == "__main__":
    pytest.main([__file__])
