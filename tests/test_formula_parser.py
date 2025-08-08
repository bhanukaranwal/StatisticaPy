# tests/test_formula_parser.py

import numpy as np
import pytest
import pandas as pd

from statisticapy.utils import formula_parser as fp

@pytest.mark.skipif(not fp._HAS_PATSY, reason="patsy not installed")
def test_parse_with_interactions_and_no_intercept():
    data = pd.DataFrame({
        'y': [1, 2, 3],
        'x1': [1, 2, 3],
        'x2': [4, 5, 6]
    })
    formula = 'y ~ x1 * x2 - 1'
    y, X = fp.parse_formula(formula, data)
    assert y.shape[0] == 3
    assert X.shape[1] == 3  # x1, x2, x1:x2 terms, no intercept

def test_basic_parse_with_intercept_and_dict():
    data = {
        'y': [1, 3, 5],
        'x1': [0, 1, 2]
    }
    formula = 'y ~ x1'
    y, X = fp.parse_formula(formula, data)
    assert y.shape[0] == 3
    assert X.shape[1] == 2  # intercept and x1

def test_no_intercept_removal():
    data = pd.DataFrame({
        'y': [1, 3],
        'x1': [0, 1]
    })
    formula = 'y ~ x1 -1'
    y, X = fp.parse_formula(formula, data)
    assert X.shape[1] == 1  # only x1, no intercept

def test_invalid_formula_raises():
    data = pd.DataFrame({'y':[1,2], 'x':[3,4]})
    with pytest.raises(ValueError):
        fp.parse_formula('invalid formula', data)

def test_parse_empty_predictors():
    data = pd.DataFrame({'y':[1,2,3]})
    formula = 'y ~ 1'
    y, X = fp.parse_formula(formula, data)
    assert y.shape[0] == 3
    assert X.shape[1] == 1  # only intercept

if __name__ == '__main__':
    pytest.main([__file__])
