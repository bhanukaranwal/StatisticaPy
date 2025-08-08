# statisticapy/utils/formula_parser.py

import numpy as np
import pandas as pd

try:
    import patsy
    _HAS_PATSY = True
except ImportError:
    _HAS_PATSY = False

class FormulaParser:
    """
    R-style formula parser to convert formula string and data into design matrices.
    
    This class wraps patsy if available; otherwise uses a minimal fallback implementation.
    
    Parameters
    ----------
    formula : str
        Model formula string, e.g., 'y ~ x1 + x2 + x1:x2 - 1'
    data : pandas.DataFrame or dict-like
        Dataset containing variables in the formula.
    """
    def __init__(self, formula, data):
        self.formula = formula
        self.data = data
    
    def parse(self):
        """
        Parse the formula and data to return response vector and design matrix.
        
        Returns
        -------
        y : ndarray
            Response variable array.
        X : ndarray
            Design matrix including intercept if specified.
        """
        if _HAS_PATSY:
            y, X = patsy.dmatrices(self.formula, self.data, return_type='dataframe')
            return np.asarray(y).ravel(), np.asarray(X)
        else:
            # Minimal fallback: support formulas like 'y ~ x1 + x2'
            # No interaction or special terms supported
            response, predictors = self.formula.split('~')
            response = response.strip()
            predictors = predictors.strip()
            
            if isinstance(self.data, dict):
                df = pd.DataFrame(self.data)
            else:
                df = self.data
            
            y = df[response].values
            
            # Remove intercept if '-1' or '+0' appears
            intercept = True
            if '-1' in predictors or '+0' in predictors:
                intercept = False
                predictors = predictors.replace('-1', '').replace('+0', '')
            
            # Split predictors by '+' ignoring spaces
            vars = [v.strip() for v in predictors.split('+') if v.strip()]
            
            X = df[vars].values if vars else np.empty((len(df), 0))
            
            if intercept:
                X = np.column_stack((np.ones(len(df)), X))
            
            return y, X

def parse_formula(formula, data):
    """
    Convenience function to parse formula string and data.
    
    Parameters
    ----------
    formula : str
        Model formula string.
    data : pandas.DataFrame or dict-like
    
    Returns
    -------
    y : ndarray
    X : ndarray
    """
    parser = FormulaParser(formula, data)
    return parser.parse()
