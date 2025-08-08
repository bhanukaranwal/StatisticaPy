# statisticapy/diagnostics/influence.py

import numpy as np
import matplotlib.pyplot as plt

def leverage(X):
    """
    Compute leverage values (diagonal of the hat matrix).
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Design matrix (with intercept if applicable).
    
    Returns
    -------
    h : ndarray of shape (n_samples,)
        Leverage values for each observation.
    """
    XtX_inv = np.linalg.pinv(X.T @ X)
    H = X @ XtX_inv @ X.T
    return np.diag(H)

def cooks_distance(X, y_true, y_pred):
    """
    Compute Cook's distance for each observation.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Design matrix (with intercept if applicable).
    y_true : ndarray of shape (n_samples,)
        True response values.
    y_pred : ndarray of shape (n_samples,)
        Predicted response values.
    
    Returns
    -------
    D : ndarray of shape (n_samples,)
        Cook's distance values for each observation.
    """
    n, p = X.shape
    residuals = y_true - y_pred
    mse = np.sum(residuals ** 2) / (n - p)
    h = leverage(X)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        D = (residuals ** 2) / (p * mse) * (h / ((1 - h) ** 2))
        # Replace NaN or infinite values (due to divide by zero or zero residuals) with zeros
        D = np.nan_to_num(D)
    return D

def dfbetas(X, y_true, y_pred):
    """
    Compute DFBETAS values for each observation and each parameter.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Design matrix (with intercept if applicable).
    y_true : ndarray of shape (n_samples,)
        True response values.
    y_pred : ndarray of shape (n_samples,)
        Predicted response values.
    
    Returns
    -------
    dfbetas_vals : ndarray of shape (n_samples, n_features)
        The DFBETAS values; how much each parameter changes when the observation is deleted.
    """
    n, p = X.shape
    residuals = y_true - y_pred
    XtX_inv = np.linalg.pinv(X.T @ X)
    h = leverage(X)
    mse = np.sum(residuals ** 2) / (n - p)
    se_beta = np.sqrt(np.diag(mse * XtX_inv))
    
    dfbetas_vals = np.zeros((n, p))
    for i in range(n):
        denom = se_beta * np.sqrt(1 - h[i])
        if denom.any() == 0:
            denom[denom == 0] = np.inf  # Avoid divide-by-zero
        numer = residuals[i] * XtX_inv @ X[i, :].T
        dfbetas_vals[i, :] = numer / denom
    return dfbetas_vals

def plot_cooks_distance(cooks_d, threshold=None, ax=None):
    """
    Plot Cook's distance values with optional threshold line.
    
    Parameters
    ----------
    cooks_d : ndarray
        Cook's distance values for observations.
    threshold : float, optional
        Threshold to highlight large influence points.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes to plot on. If None, creates new.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    n = len(cooks_d)
    ax.stem(range(n), cooks_d, basefmt=" ")
    ax.set_xlabel("Observation index")
    ax.set_ylabel("Cook's distance")
    ax.set_title("Cook's Distance Plot")
    
    if threshold is not None:
        ax.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
        ax.legend()
    
    return ax
