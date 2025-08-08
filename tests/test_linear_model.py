# tests/test_linear_model.py

import numpy as np
import pytest
from statisticapy.models.linear_model import LinearRegression

def test_linear_regression_fit_predict():
    X = np.array([[1, 2], [2, 3], [4, 5], [3, 2]])
    y = np.array([3, 5, 9, 6])
    
    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)
    
    # Check prediction shape
    assert preds.shape == (X.shape[0],)
    # Check predictions are reasonably close to y
    assert np.allclose(preds, y, atol=1e-5)

def test_predict_before_fit_raises():
    X = np.array([[1, 2], [3, 4]])
    model = LinearRegression()
    with pytest.raises(RuntimeError):
        model.predict(X)

def test_singular_matrix_raises():
    # Two identical rows cause singularity in X'X
    X = np.array([[1, 2], [1, 2], [3, 4]])
    y = np.array([3, 3, 7])
    model = LinearRegression()
    with pytest.raises(ValueError):
        model.fit(X, y)

if __name__ == "__main__":
    pytest.main([__file__])
