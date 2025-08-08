# statisticapy/diagnostics/hypothesis_tests.py

import numpy as np
from scipy import stats

def one_sample_ttest(data, popmean=0, alternative='two-sided'):
    """
    Perform a one-sample t-test.
    
    Parameters
    ----------
    data : array-like
        Sample data.
    popmean : float, optional
        The population mean to test against.
    alternative : {'two-sided', 'less', 'greater'}, default 'two-sided'
        Defines the alternative hypothesis.
    
    Returns
    -------
    t_stat : float
        The calculated t-statistic.
    p_value : float
        The p-value for the test.
    """
    data = np.asarray(data)
    n = len(data)
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(n)
    t_stat = (mean - popmean) / std_err
    df = n - 1
    
    if alternative == 'two-sided':
        p_value = 2 * stats.t.sf(np.abs(t_stat), df)
    elif alternative == 'less':
        p_value = stats.t.cdf(t_stat, df)
    elif alternative == 'greater':
        p_value = stats.t.sf(t_stat, df)
    else:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")
    
    return t_stat, p_value

def two_sample_ttest(data1, data2, equal_var=True, alternative='two-sided'):
    """
    Perform a two-sample t-test.
    
    Parameters
    ----------
    data1, data2 : array-like
        Sample data from two independent groups.
    equal_var : bool, default True
        If True, perform standard independent 2 sample test that assumes equal population variances.
        If False, perform Welch’s t-test.
    alternative : {'two-sided', 'less', 'greater'}, default 'two-sided'
        Defines the alternative hypothesis.
    
    Returns
    -------
    t_stat : float
    p_value : float
    """
    t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
    
    if alternative == 'less':
        if t_stat >= 0:
            p_value = 1 - p_value / 2
        else:
            p_value = p_value / 2
    elif alternative == 'greater':
        if t_stat <= 0:
            p_value = 1 - p_value / 2
        else:
            p_value = p_value / 2
    elif alternative != 'two-sided':
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")
    
    return t_stat, p_value

def chi2_test(observed, expected=None):
    """
    Perform Chi-square test for goodness of fit or independence.
    
    Parameters
    ----------
    observed : array-like
        Observed frequencies (contingency table or 1D array).
    expected : array-like, optional
        Expected frequencies under the null hypothesis.
    
    Returns
    -------
    chi2_stat : float
        Chi-square test statistic
    p_value : float
        p-value of the test
    dof : int
        Degrees of freedom
    """
    observed = np.asarray(observed)
    if expected is not None:
        expected = np.asarray(expected)
    
    chi2_stat, p_value, dof, _ = stats.chi2_contingency(observed, correction=False)
    
    return chi2_stat, p_value, dof

def one_way_anova(*groups):
    """
    Perform one-way ANOVA test.
    
    Parameters
    ----------
    groups : array-like
        Variable number of groups, each an array of sample observations.
    
    Returns
    -------
    F_stat : float
        The computed F-statistic.
    p_value : float
        The p-value of the test.
    """
    F_stat, p_value = stats.f_oneway(*groups)
    return F_stat, p_value

def wilcoxon_signed_rank(x, y=None, alternative='two-sided'):
    """
    Perform Wilcoxon signed-rank test.
    
    Parameters
    ----------
    x, y : array-like
        Paired samples to compare or single sample when y is None.
    alternative : {'two-sided', 'less', 'greater'}, default 'two-sided'
    
    Returns
    -------
    stat : float
        The test statistic.
    p_value : float
        The p-value for the test.
    """
    stat, p_value = stats.wilcoxon(x, y=y, alternative=alternative)
    return stat, p_value

def ks_test(data1, data2=None, alternative='two-sided'):
    """
    Perform Kolmogorov-Smirnov test.
    
    Parameters
    ----------
    data1 : array-like
        First sample data.
    data2 : array-like, optional
        Second sample data. If None, compares data1 to a standard uniform distribution.
    alternative : {'two-sided', 'less', 'greater'}, default 'two-sided'
    
    Returns
    -------
    stat : float
        KS test statistic.
    p_value : float
        p-value for the test.
    """
    if data2 is None:
        data2 = 'uniform'
    stat, p_value = stats.ks_2samp(data1, data2) if isinstance(data2, (np.ndarray, list)) else stats.kstest(data1, data2, alternative=alternative)
    return stat, p_value
