# tests/test_robust_stats.py

import numpy as np
import pytest
from statisticapy.diagnostics import robust_stats as rs

def test_robust_skewness_basic():
    data = np.array([1, 2, 3, 4, 5])
    skewness = rs.robust_skewness(data)
    assert isinstance(skewness, float)

def test_robust_skewness_constant():
    data = np.array([5, 5, 5, 5])
    skewness = rs.robust_skewness(data)
    assert skewness == 0.0

def test_robust_kurtosis_basic():
    data = np.array([1, 2, 3, 4, 5])
    kurtosis = rs.robust_kurtosis(data)
    assert isinstance(kurtosis, float)

def test_robust_kurtosis_constant():
    data = np.array([7, 7, 7])
    kurtosis = rs.robust_kurtosis(data)
    assert kurtosis == 0.0

def test_median_absolute_deviation_normal_and_none():
    data = np.array([1, 2, 2, 4, 6, 9])
    mad_normal = rs.median_absolute_deviation(data, scale='normal')
    mad_none = rs.median_absolute_deviation(data, scale=None)
    assert mad_normal >= 0
    assert mad_none >= 0
    assert mad_normal >= mad_none  # Because normal scaling >1

def test_modified_z_score_and_outlier_detection():
    data = np.array([10, 12, 12, 13, 12, 14, 100])
    z_scores = rs.modified_z_score(data)
    assert z_scores.shape == data.shape
    outliers = rs.detect_outliers_modified_z_score(data, threshold=3.5)
    assert isinstance(outliers, np.ndarray)
    assert outliers[-1] == True  # The last value 100 should be identified as outlier

def test_modified_z_score_constant_data():
    data = np.array([5, 5, 5, 5])
    z_scores = rs.modified_z_score(data)
    assert np.all(z_scores == 0)

if __name__ == "__main__":
    pytest.main([__file__])
