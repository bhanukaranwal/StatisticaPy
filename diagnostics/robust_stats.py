# statisticapy/diagnostics/robust_stats.py

import numpy as np
from scipy.stats import median_abs_deviation

def robust_skewness(x):
    """
    Calculate a robust measure of skewness based on medcouple.
    
    Parameters
    ----------
    x : array-like
        Input data.
    
    Returns
    -------
    float
        Robust skewness estimator.
    """
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    # Use medcouple implementation if available in scipy.stats (>=1.9.0)
    try:
        from scipy.stats import medcouple
        return medcouple(x)
    except ImportError:
        # Fallback: simple skewness as (mean - median)/std robust approx
        median = np.median(x)
        mean = np.mean(x)
        std = np.std(x, ddof=1)
        if std == 0:
            return 0.0
        return 3 * (mean - median) / std

def robust_kurtosis(x):
    """
    Calculate a robust measure of kurtosis.
    
    Parameters
    ----------
    x : array-like
        Input data.
    
    Returns
    -------
    float
        Robust kurtosis estimator.
    """
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    # Using adjusted Fisher-Pearson kurtosis with robust scale
    
    q75, q25 = np.percentile(x, [75 ,25])
    iqr = q75 - q25
    mad = median_abs_deviation(x, scale='normal')
    
    if mad == 0:
        return 0.0
    
    # Robust kurtosis proxy: ratio of iqr to mad scaled and adjusted
    return (iqr / mad) ** 2

def median_absolute_deviation(x, scale='normal'):
    """
    Compute the median absolute deviation (MAD), a robust scale estimator.
    
    Parameters
    ----------
    x : array-like
        Input data.
    scale : {'normal', None}, default 'normal'
        If 'normal', scale MAD by approximately 1.4826 to be consistent with std
        estimation for normal distribution.
    
    Returns
    -------
    float
        MAD estimate.
    """
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    mad = median_abs_deviation(x, scale=scale)
    return mad

def modified_z_score(x):
    """
    Compute modified Z-scores for outlier detection.
    
    Parameters
    ----------
    x : array-like
        Input data.
    
    Returns
    -------
    z_scores : ndarray
        Modified Z-scores.
    """
    x = np.asarray(x)
    median = np.median(x)
    mad = median_abs_deviation(x, scale='normal')
    if mad == 0:
        # Avoid division by zero
        return np.zeros_like(x)
    z_scores = 0.6745 * (x - median) / mad
    return z_scores

def detect_outliers_modified_z_score(x, threshold=3.5):
    """
    Detect outliers based on modified Z-score method.
    
    Parameters
    ----------
    x : array-like
        Input data.
    threshold : float, default=3.5
        Threshold above which a data point is considered an outlier.
    
    Returns
    -------
    outliers : ndarray of bool
        Boolean array where True indicates an outlier.
    """
    z_scores = modified_z_score(x)
    return np.abs(z_scores) > threshold
