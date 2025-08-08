# tests/test_residuals.py

import numpy as np
import matplotlib.pyplot as plt
import pytest

from statisticapy.diagnostics import residuals as res

def test_raw_residuals():
    y_true = np.array([3, 4, 5])
    y_pred = np.array([2.5, 4.0, 5.5])
    residuals = res.raw_residuals(y_true, y_pred)
    expected = y_true - y_pred
    assert np.allclose(residuals, expected)

def test_standardized_residuals():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.1, 2.9, 3.8, 5.2])
    std_res = res.standardized_residuals(y_true, y_pred)
    assert np.abs(np.mean(std_res)) < 1e-10
    assert np.isclose(np.std(std_res, ddof=1), 1.0)

def test_studentized_residuals():
    # Create simple linear data with intercept
    X = np.column_stack((np.ones(5), np.arange(1, 6)))
    y_true = np.array([2, 3, 4, 5, 6])  # perfect fit line y = x +1
    y_pred = np.array([2, 3, 4, 5, 6])
    
    stud_res = res.studentized_residuals(X, y_true, y_pred)
    # Since perfect fit, residuals are zero, studentized residuals should be zero or nan (handle carefully)
    assert stud_res.shape == (5,)
    # May have divide-by-zero issues, but should not error out

def test_plot_residuals_vs_fitted_smoke():
    y_true = np.array([2, 4, 6, 8])
    y_pred = np.array([2.1, 3.9, 6.1, 8.2])
    ax = res.plot_residuals_vs_fitted(y_true, y_pred)
    assert ax.get_xlabel() == "Fitted values"
    plt.close(ax.figure)

def test_breusch_pagan_test_basic():
    np.random.seed(0)
    X = np.column_stack((np.ones(100), np.random.normal(size=100)))
    y = 2 + 3 * X[:,1] + np.random.normal(size=100)
    y_pred = 2 + 3 * X[:,1]
    
    lm_stat, p_value = res.breusch_pagan_test(X, y, y_pred)
    # lm_stat should be non-negative
    assert lm_stat >= 0
    # p_value bounded between 0 and 1
    assert 0 <= p_value <= 1

def test_durbin_watson_statistic():
    residuals = np.array([1, 0, -1, 0, 1])
    dw_stat = res.durbin_watson(residuals)
    assert isinstance(dw_stat, float)
    # DW statistic ranges between 0 and 4 for residuals
    assert 0 <= dw_stat <= 4

if __name__ == "__main__":
    pytest.main([__file__])
