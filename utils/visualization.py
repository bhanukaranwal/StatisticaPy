# statisticapy/utils/visualization.py

import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False

def set_style(style='default'):
    """
    Set plotting style.
    
    Parameters
    ----------
    style : str, default 'default'
        Can be 'default', 'seaborn', or any style recognized by plt.style.use.
    """
    if style == 'seaborn' and _HAS_SEABORN:
        sns.set_style('whitegrid')
    else:
        plt.style.use(style)

def plot_residuals_vs_fitted(y_true, y_pred, ax=None):
    """
    Plot residuals versus fitted values.
    
    Parameters
    ----------
    y_true: array-like
        Observed values.
    y_pred: array-like
        Predicted values.
    ax: matplotlib.axes.Axes, optional
    
    Returns
    -------
    ax: matplotlib.axes.Axes
    """
    import statisticapy.diagnostics.residuals as residuals

    if ax is None:
        fig, ax = plt.subplots()
    residual_vals = residuals.raw_residuals(y_true, y_pred)
    ax.scatter(y_pred, residual_vals, alpha=0.7, edgecolor='k' if _HAS_SEABORN else None)
    ax.axhline(0, linestyle='--', color='r')
    ax.set_title('Residuals vs Fitted')
    ax.set_xlabel('Fitted values')
    ax.set_ylabel('Residuals')
    return ax

def qq_plot(data, dist='norm', ax=None, line=True):
    """
    Generate a Quantile-Quantile plot to compare data distribution to a theoretical distribution.
    
    Parameters
    ----------
    data : array-like
        Data sample.
    dist : str or scipy.stats distribution object, default 'norm'
        The theoretical distribution to compare against.
    ax : matplotlib.axes.Axes, optional
    
    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    import scipy.stats as stats
    
    if ax is None:
        fig, ax = plt.subplots()
    
    stats.probplot(data, dist=dist, plot=ax)
    if line:
        # The stats.probplot adds a line automatically, but we can customize if needed.
        pass
    ax.set_title('Q-Q Plot')
    return ax

def histogram(data, bins=30, ax=None, kde=False):
    """
    Plot histogram of data with optional KDE curve.
    
    Parameters
    ----------
    data : array-like
        Data sample.
    bins : int, default 30
        Number of bins.
    ax : matplotlib.axes.Axes, optional
    kde : bool, default False
        Whether to overlay Kernel Density Estimate (requires Seaborn)
    
    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    if kde and _HAS_SEABORN:
        import seaborn as sns
        sns.histplot(data, bins=bins, kde=True, ax=ax)
    else:
        ax.hist(data, bins=bins, alpha=0.7, edgecolor='black')
    ax.set_title('Histogram')
    return ax

def scatter_plot(x, y, ax=None, hue=None):
    """
    Create a scatter plot with optional hue grouping.
    
    Parameters
    ----------
    x : array-like
        X-axis values.
    y : array-like
        Y-axis values.
    ax : matplotlib.axes.Axes, optional
    hue : array-like or None
        Grouping variable for color coding.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    if _HAS_SEABORN:
        import seaborn as sns
        sns.scatterplot(x=x, y=y, hue=hue, ax=ax)
    else:
        sc = ax.scatter(x, y, c=hue if hue is not None else 'b', alpha=0.7)
        if hue is not None:
            ax.legend(*sc.legend_elements(), title="Groups")
    ax.set_title('Scatter Plot')
    return ax
