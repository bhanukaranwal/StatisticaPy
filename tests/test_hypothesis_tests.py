# tests/test_hypothesis_tests.py

import numpy as np
import pytest
from statisticapy.diagnostics import hypothesis_tests as ht

def test_one_sample_ttest_two_sided():
    data = np.array([1.1, 2.3, 1.9, 2.2, 1.8])
    t_stat, p_val = ht.one_sample_ttest(data, popmean=2.0)
    assert isinstance(t_stat, float)
    assert 0 <= p_val <= 1

def test_one_sample_ttest_alternatives():
    data = np.array([2.1, 2.3, 2.2, 2.5])
    for alt in ['two-sided', 'less', 'greater']:
        t_stat, p_val = ht.one_sample_ttest(data, popmean=2.0, alternative=alt)
        assert isinstance(p_val, float)
    with pytest.raises(ValueError):
        ht.one_sample_ttest(data, popmean=2.0, alternative='invalid')

def test_two_sample_ttest_equal_var_and_welch():
    data1 = np.random.normal(0, 1, 30)
    data2 = np.random.normal(0.1, 1, 30)
    # Equal variances
    t_stat, p_val = ht.two_sample_ttest(data1, data2, equal_var=True)
    assert 0 <= p_val <= 1
    # Welch's test
    t_stat, p_val = ht.two_sample_ttest(data1, data2, equal_var=False)
    assert 0 <= p_val <= 1
    # Alternative hypotheses
    for alt in ['two-sided', 'less', 'greater']:
        t_stat, p_val = ht.two_sample_ttest(data1, data2, equal_var=True, alternative=alt)
    with pytest.raises(ValueError):
        ht.two_sample_ttest(data1, data2, equal_var=True, alternative='invalid')

def test_chi2_test():
    observed = np.array([[10, 20], [20, 40]])
    chi2_stat, p_val, dof = ht.chi2_test(observed)
    assert chi2_stat >= 0
    assert 0 <= p_val <= 1
    assert isinstance(dof, int)

def test_one_way_anova():
    group1 = np.random.normal(0, 1, 20)
    group2 = np.random.normal(0.5, 1, 20)
    group3 = np.random.normal(1.0, 1, 20)
    F_stat, p_val = ht.one_way_anova(group1, group2, group3)
    assert F_stat >= 0
    assert 0 <= p_val <= 1

def test_wilcoxon_signed_rank():
    x = np.array([1.2, 2.3, 3.1, 4.8])
    y = np.array([1.1, 2.0, 2.9, 5.0])
    stat, p_val = ht.wilcoxon_signed_rank(x, y)
    assert 0 <= p_val <= 1

def test_ks_test_two_sample_and_one_sample():
    data1 = np.random.normal(0, 1, 50)
    data2 = np.random.normal(0.1, 1, 50)
    stat, p_val = ht.ks_test(data1, data2)
    assert 0 <= p_val <= 1
    
    # One sample KS test against uniform distribution
    uniform_sample = np.random.uniform(0, 1, 50)
    stat, p_val = ht.ks_test(uniform_sample, data2=None)
    assert 0 <= p_val <= 1

if __name__ == '__main__':
    pytest.main([__file__])
