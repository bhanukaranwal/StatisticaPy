# statisticapy/models/linear_model.py
def predict(self, X):
    """
    Predict using the linear model.

    Parameters
    ----------
    X : array-like or DataFrame of shape (n_samples, n_features)
        Samples; if model was fit with formula, can be a pandas DataFrame compatible
        with the formula.

    Returns
    -------
    y_pred : ndarray of shape (n_samples,)
        Predicted values
    """
    if not self.fitted:
        raise RuntimeError("You must fit the model before prediction.")

    if self.formula is not None:
        # Expect DataFrame input for consistent parsing
        import pandas as pd
        if not isinstance(X, pd.DataFrame):
            raise TypeError("When fitted with formula, predict expects pandas DataFrame input.")

        # Use formula parser to generate design matrix (without response variable)
        # Trick: parse formula without response to get design matrix for predictors only
        from ..utils.formula_parser import FormulaParser

        # Extract formula RHS only: e.g., 'y ~ x1 + x2' --> 'x1 + x2'
        rhs_formula = self.formula.split('~', 1)[1].strip()
        # Build formula with no response, but intercept respected
        design_parser = FormulaParser(f"~ {rhs_formula}", X)
        _, X_design = design_parser.parse()

        # Prediction with design matrix - assume intercept included if formula specified
        return X_design @ self.params_
    else:
        X = np.asarray(X)
        if self.fit_intercept:
            X = np.column_stack((np.ones(X.shape[0]), X))
        return X @ self.params_

import numpy as np
from ..core import BaseModel
from ..utils.formula_parser import parse_formula

class LinearRegression(BaseModel):
    """
    Ordinary Least Squares (OLS) Linear Regression
    
    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. Ignored if formula is given.
    formula : str, optional
        Model formula string e.g. 'y ~ x1 + x2'.
        If provided, fit() expects data parameter.
    
    Attributes
    ----------
    params_ : ndarray of shape (n_features,) or (n_features + 1,)
        Estimated coefficients including intercept if fit_intercept=True.
    fitted : bool
        Whether the model has been fitted.
    feature_names_ : list of str
        Names of features in the design matrix after parsing formula.
    """
    def __init__(self, fit_intercept=True, formula=None):
        super().__init__()
        self.fit_intercept = fit_intercept
        self.formula = formula
        self.feature_names_ = None
    
    def fit(self, X=None, y=None, data=None):
        """
        Fit the linear model.
        
        If formula is set, X and y should not be provided. Instead provide data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), optional
            Feature data.
        y : array-like of shape (n_samples,), optional
            Target values.
        data : DataFrame or dict-like, optional
            Dataset used with formula if formula is provided.
        
        Returns
        -------
        self : object
        """
        if self.formula is not None:
            if data is None:
                raise ValueError("data must be provided when formula is used")
            y, X = parse_formula(self.formula, data)
            # Intercept handled automatically by formula parser
        else:
            if X is None or y is None:
                raise ValueError("X and y must be provided if formula is not used")
            X = np.asarray(X)
            y = np.asarray(y)
            if self.fit_intercept:
                X = np.column_stack((np.ones(X.shape[0]), X))
        
        # Save feature names if formula used
        if self.formula is not None and hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)
        else:
            # If no formula, feature names unknown or numeric indices
            self.feature_names_ = [f"x{i}" for i in range(X.shape[1])]

        XtX = X.T @ X
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            raise ValueError("Singular matrix detected during fit; possibly multicollinearity.")
        
        Xty = X.T @ y
        self.params_ = XtX_inv @ Xty
        self.fitted = True
        return self
    
    def predict(self, X):
        """
        Predict using the linear model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values
        """
        if not self.fitted:
            raise RuntimeError("You must fit the model before prediction.")
        
        X = np.asarray(X)
        if self.formula is None and self.fit_intercept:
            X = np.column_stack((np.ones(X.shape[0]), X))
        elif self.formula is not None:
            # If formula was used, attempt to parse input X similarly
            # For prediction, users should supply data as DataFrame
            raise NotImplementedError("Prediction with formula requires DataFrame input and formula parsing is not implemented for predict.")
        
        return X @ self.params_
