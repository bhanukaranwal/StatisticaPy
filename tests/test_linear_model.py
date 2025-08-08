import numpy as np
import pytest
import pandas as pd
from statisticapy.models.linear_model import LinearRegression

def test_predict_with_formula_and_dataframe():
    data = pd.DataFrame({
        'y': [1, 3, 5, 7],
        'x1': [0, 1, 2, 3],
        'x2': [1, 2, 3, 4]
    })
    formula = 'y ~ x1 + x2'
    
    model = LinearRegression(formula=formula)
    model.fit(data=data)
    
    # Correct DataFrame input
    new_data = pd.DataFrame({
        'x1': [4, 5],
        'x2': [5, 6]
    })
    preds = model.predict(new_data)
    assert preds.shape == (2,)
    
    # Passing non-DataFrame input should raise TypeError
    with pytest.raises(TypeError):
        model.predict(np.array([[4, 5], [5, 6]]))

def test_predict_without_fit_raises():
    model = LinearRegression()
    with pytest.raises(RuntimeError):
        model.predict(np.array([[1, 2]]))
