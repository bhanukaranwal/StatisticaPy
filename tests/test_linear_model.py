# tests/test_linear_model.py

import numpy as np
import pytest
import pandas as pd
from statisticapy.models.linear_model import LinearRegression

def test_linear_regression_fit_predict():
    X = np.array([[1, 2], [2, 3], [4, 5], [3, 2]])
    y = np.array([3, 5, 9, 6])
    
    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)
    
    assert preds.shape == (X.shape[0],)
    assert np.allclose(preds, y, atol=1e-5)

def test_predict_before_fit_raises():
    X = np.array([[1, 2], [3, 4]])
    model = LinearRegression()
    with pytest.raises(RuntimeError):
        model.predict(X)

def test_singular_matrix_raises():
    X = np.array([[1, 2], [1, 2], [3, 4]])
    y = np.array([3, 3, 7])
    model = LinearRegression()
    with pytest.raises(ValueError):
        model.fit(X, y)

def test_fit_with_formula_and_dataframe():
    import pandas as pd
    data = pd.DataFrame({
        'y': [1, 3, 5, 7],
        'x1': [0, 1, 2, 3],
        'x2': [1, 2, 3, 4]
    })
    formula = 'y ~ x1 + x2'
    model = LinearRegression(formula=formula)
    model.fit(data=data)
    preds = model.predict(np.array([[1, 2], [3, 4]]))  # Predict with numeric array & no intercept auto add
    
    assert preds.shape == (2,)
    # Predict with formula raises not implemented error
    with pytest.raises(NotImplementedError):
        model.predict(data)

def test_fit_formula_requires_data():
    model = LinearRegression(formula='y ~ x1 + x2')
    with pytest.raises(ValueError):
        model.fit()  # data not provided with formula

if __name__ == "__main__":
    pytest.main([__file__])
