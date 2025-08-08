# tests/test_generalized_linear_model.py

import numpy as np
import pytest
import pandas as pd
from statisticapy.models.generalized_linear_model import GeneralizedLinearModel, LogisticRegression, PoissonRegression
from statisticapy.models.generalized_linear_model import Gaussian, Binomial, Poisson

def test_glm_fit_predict_numeric():
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([1, 2, 3, 4])
    glm = GeneralizedLinearModel(family=Gaussian(), fit_intercept=True)
    glm.fit(X, y)
    preds = glm.predict(X)
    assert preds.shape == (X.shape[0],)

def test_glm_fit_predict_formula():
    data = pd.DataFrame({
        'y': [0, 1, 0, 1, 1],
        'x1': [1, 2, 3, 4, 5],
        'x2': [5, 4, 3, 2, 1]
    })
    formula = 'y ~ x1 + x2'
    
    glm = GeneralizedLinearModel(family=Binomial(), formula=formula)
    glm.fit(data=data)
    # Prediction with numeric array should work if intercept handled
    X_new = np.array([[6, 0], [7, -1]])
    preds = glm.predict(X_new)
    assert preds.shape == (2,)

def test_logistic_regression_with_formula():
    data = pd.DataFrame({
        'y': [0, 1, 0, 1, 1],
        'x1': [0.5, 1.2, 1.8, 3.0, 2.9],
        'x2': [1.2, 3.1, 2.8, 0.5, 2.7]
    })
    formula = 'y ~ x1 + x2'
    model = LogisticRegression(formula=formula)
    model.fit(data=data)
    preds = model.predict(np.array([[2, 1], [1, 0.5]]))
    assert preds.shape == (2,)
    assert np.all((preds >= 0) & (preds <= 1))  # Probabilities

def test_poisson_regression_with_formula():
    data = pd.DataFrame({
        'y': [1, 3, 7, 6, 8],
        'x1': [1, 2, 3, 4, 5],
        'x2': [0.5, 1.5, 3.5, 0.5, 2.0]
    })
    formula = 'y ~ x1 + x2'
    model = PoissonRegression(formula=formula)
    model.fit(data=data)
    preds = model.predict(np.array([[6, 1], [7, 2]]))
    assert preds.shape == (2,)
    assert np.all(preds >= 0)

def test_fit_with_formula_requires_data():
    glm = GeneralizedLinearModel(formula='y ~ x1 + x2')
    with pytest.raises(ValueError):
        glm.fit()

def test_predict_before_fit_raises():
    model = LogisticRegression()
    X = np.array([[1, 2]])
    with pytest.raises(RuntimeError):
        model.predict(X)

def test_predict_with_formula_not_implemented():
    data = pd.DataFrame({
        'y': [0, 1],
        'x1': [1, 2]
    })
    formula = 'y ~ x1'
    model = LogisticRegression(formula=formula)
    model.fit(data=data)
    with pytest.raises(NotImplementedError):
        model.predict(data)

if __name__ == "__main__":
    pytest.main([__file__])
