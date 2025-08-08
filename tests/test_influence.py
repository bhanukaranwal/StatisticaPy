# tests/test_influence.py

import numpy as np
import pytest
import matplotlib.pyplot as plt

from statisticapy.diagnostics import influence as infl

def test_leverage_values_and_shape():
    np.random.seed(0)
    X = np.column_stack((np.ones(5), np.arange(5)))
    h = infl.leverage(X)
    assert h.shape == (X.shape[0],)
    assert np.all(h >= 0) and np.all(h <= 1)
    # Sum of leverage values equals number of parameters (degrees of freedom)
    assert np.isclose(np.sum(h), X.shape[1], atol=1e-6)

def test_cooks_distance_values_and_shape():
    np.random.seed(1)
    X = np.column_stack((np.ones(5), np.arange(5)))
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.05])
    cooks_d = infl.cooks_distance(X, y_true, y_pred)
    assert cooks_d.shape == (X.shape[0],)
    assert np.all(cooks_d >= 0)

def test_dfbetas_values_and_shape():
    np.random.seed(2)
    X = np.column_stack((np.ones(6), np.arange(6)))
    y_true = 3 + 2 * X[:,1] + np.random.normal(size=6)
    y_pred = 3 + 2 * X[:,1]
    dfbetas_vals = infl.dfbetas(X, y_true, y_pred)
    assert dfbetas_vals.shape == (X.shape[0], X.shape[1])
    # Values should be finite numbers
    assert np.all(np.isfinite(dfbetas_vals))

def test_plot_cooks_distance_smoke():
    cooks_d = np.array([0.1, 0.05, 0.2, 0.5, 0.05])
    ax = infl.plot_cooks_distance(cooks_d, threshold=0.3)
    assert ax.get_title() == "Cook's Distance Plot"
    plt.close(ax.figure)

if __name__ == "__main__":
    pytest.main([__file__])
