# tests/test_visualization.py

import numpy as np
import pytest
import matplotlib.pyplot as plt

from statisticapy.utils import visualization as viz

def test_set_style():
    # Test valid style selections
    viz.set_style('default')
    if hasattr(viz, '_HAS_SEABORN') and viz._HAS_SEABORN:
        viz.set_style('seaborn')
    # Test unknown style falls back without error
    viz.set_style('nonexistent-style')  # Should not raise error but may warn

def test_plot_residuals_vs_fitted_smoke():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([0.9, 2.1, 2.9, 4.2])
    ax = viz.plot_residuals_vs_fitted(y_true, y_pred)
    assert ax.get_xlabel() == "Fitted values"
    plt.close(ax.figure)

def test_qq_plot_smoke():
    data = np.random.normal(size=100)
    ax = viz.qq_plot(data)
    assert ax.get_title() == "Q-Q Plot"
    plt.close(ax.figure)

def test_histogram_with_and_without_kde():
    data = np.random.normal(size=100)
    ax1 = viz.histogram(data, kde=False)
    assert ax1.get_title() == "Histogram"
    plt.close(ax1.figure)
    
    if hasattr(viz, '_HAS_SEABORN') and viz._HAS_SEABORN:
        ax2 = viz.histogram(data, kde=True)
        assert ax2.get_title() == "Histogram"
        plt.close(ax2.figure)

def test_scatter_plot_with_and_without_hue():
    x = np.random.rand(10)
    y = np.random.rand(10)
    ax1 = viz.scatter_plot(x, y)
    assert ax1.get_title() == "Scatter Plot"
    plt.close(ax1.figure)
    
    hue = np.random.choice(['A', 'B'], size=10)
    ax2 = viz.scatter_plot(x, y, hue=hue)
    assert ax2.get_title() == "Scatter Plot"
    plt.close(ax2.figure)

if __name__ == "__main__":
    pytest.main([__file__])
