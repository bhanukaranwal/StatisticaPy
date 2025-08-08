import numpy as np
import pytest
import pandas as pd
from statisticapy.models.generalized_linear_model import LogisticRegression, PoissonRegression

def test_predict_with_formula_and_dataframe_logistic():
    data = pd.DataFrame({
        'y': [0, 1, 0, 1],
        'x1': [1, 2, 3, 4],
        'x2': [4, 3, 2, 1]
    })
    formula = 'y ~ x1 + x2'
    
    model = LogisticRegression(formula=formula)
    model.fit(data=data)
    
    new_data = pd.DataFrame({
        'x1': [5, 6],
        'x2': [0, -1]
    })
    
    preds = model.predict(new_data)
    assert preds.shape == (2,)
    assert np.all((preds >= 0) & (preds <= 1))
    
    # Non-DataFrame input should raise TypeError
    with pytest.raises(TypeError):
        model.predict(np.array([[5, 0], [6, -1]]))

def test_predict_with_formula_and_dataframe_poisson():
    data = pd.DataFrame({
        'y': [1, 2, 3, 4],
        'x1': [1, 3, 5, 7],
        'x2': [2, 4, 6, 8]
    })
    formula = 'y ~ x1 + x2'
    
    model = PoissonRegression(formula=formula)
    model.fit(data=data)
    
    new_data = pd.DataFrame({
        'x1': [9, 11],
        'x2': [10, 12]
    })
    
    preds = model.predict(new_data)
    assert preds.shape == (2,)
    assert np.all(preds >= 0)
    
    with pytest.raises(TypeError):
        model.predict(np.array([[9, 10], [11, 12]]))

def test_predict_without_fit_raises():
    model = LogisticRegression()
    with pytest.raises(RuntimeError):
        model.predict(np.array([[1, 2]]))
