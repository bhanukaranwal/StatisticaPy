# statisticapy/diagnostics/residuals.py

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def raw_residuals(y_true, y_pred):
    """
    Compute raw residuals: observed minus predicted.
    
    Parameters
    ----------
    y_true : array-like
        True observed values.
    y_pred : array-like
        Predicted values.
    
    Returns
    -------
    residuals : ndarray
        Raw residuals.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return y_true - y_pred

def standardized_residuals(y_true, y_pred):
    """
    Compute standardized residuals: residuals divided by their estimated std deviation.
    
    Parameters
    ----------
    y_true : array-like
        True observed values.
    y_pred : array-like
        Predicted values.
    
    Returns
    -------
    std_residuals : ndarray
        Standardized residuals.
    """
    res = raw_residuals(y_true, y_pred)
    std_res = (res - np.mean(res)) / np.std(res, ddof=1)
    return std_res

def studentized_residuals(X, y_true, y_pred):
    """
    Compute externally studentized residuals.
    
    Parameters
    ----------
    X : ndarray
        Design matrix (including intercept if applicable), shape (n_samples, n_features)
    y_true : ndarray
        True values.
    y_pred : ndarray
        Predicted values.
    
    Returns
    -------
    studentized_res : ndarray
        Studentized residuals.
    """
    X = np.asarray(X)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    residuals = y_true - y_pred
    n, p = X.shape
    mse = np.sum(residuals**2) / (n - p)
    # Hat matrix diagonal H = diag(X(X'X)^{-1}X')
    XtX_inv = np.linalg.pinv(X.T @ X)
    H = np.sum((X @ XtX_inv) * X, axis=1)
    
    denom = np.sqrt(mse * (1 - H))
    studentized = residuals / denom
    return studentized

def plot_residuals_vs_fitted(y_true, y_pred, ax=None):
    """
    Plot residuals versus fitted values.
    
    Parameters
    ----------
    y_true : array-like
        Observed values.
    y_pred : array-like
        Predicted values.
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot. Creates new if None.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    residuals = raw_residuals(y_true, y_pred)
    ax.scatter(y_pred, residuals, alpha=0.7)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs. Fitted Values")
    return ax

def breusch_pagan_test(X, y_true, y_pred):
    """
    Perform Breusch-Pagan test for heteroscedasticity.
    
    Parameters
    ----------
    X : ndarray
        Design matrix (including intercept if applicable).
    y_true : ndarray
        Observed values.
    y_pred : ndarray
        Predicted values.
    
    Returns
    -------
    lm_stat : float
        Lagrange multiplier statistic.
    p_value : float
        p-value for the test.
    """
    residuals = y_true - y_pred
    n = X.shape[0]
    rss = np.sum(residuals**2)
    
    # Compute squared residuals scaled by residual variance
    sigma2 = rss / n
    e2 = residuals**2 / sigma2
    
    # Regress e2 on X
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ e2)
    e2_hat = X @ beta
    ssr = np.sum((e2_hat - e2.mean())**2)
    
    lm_stat = 0.5 * ssr
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(lm_stat, X.shape[1] - 1)
    
    return lm_stat, p_value

def durbin_watson(residuals):
    """
    Compute Durbin-Watson statistic for autocorrelation of residuals.
    
    Parameters
    ----------
    residuals : array-like
        Residuals from regression.
    
    Returns
    -------
    dw_stat : float
        Durbin-Watson statistic.
    """
    residuals = np.asarray(residuals)
    diff = np.diff(residuals)
    dw_stat = np.sum(diff**2) / np.sum(residuals**2)
    return dw_stat
