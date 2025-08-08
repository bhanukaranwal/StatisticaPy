# examples/glm_formula_example.py

import pandas as pd
from statisticapy.models.generalized_linear_model import LogisticRegression, PoissonRegression

def logistic_regression_formula_example():
    data = pd.DataFrame({
        'y': [0, 1, 0, 1, 1],
        'x1': [0.5, 1.2, 1.8, 3.0, 2.9],
        'x2': [1.2, 3.1, 2.8, 0.5, 2.7]
    })
    formula = 'y ~ x1 + x2'
    
    model = LogisticRegression(formula=formula)
    model.fit(data=data)
    print("Logistic Regression fitted parameters:")
    print(model.params_)
    
    # Predict on new numeric data (with features only, no intercept column)
    X_new = [[2, 1], [1, 0.5]]
    preds = model.predict(X_new)
    print("Predicted probabilities:")
    print(preds)

def poisson_regression_formula_example():
    data = pd.DataFrame({
        'y': [1, 3, 7, 6, 8],
        'x1': [1, 2, 3, 4, 5],
        'x2': [0.5, 1.5, 3.5, 0.5, 2.0]
    })
    formula = 'y ~ x1 + x2'
    
    model = PoissonRegression(formula=formula)
    model.fit(data=data)
    print("Poisson Regression fitted parameters:")
    print(model.params_)
    
    X_new = [[6, 1], [7, 2]]
    preds = model.predict(X_new)
    print("Predicted means:")
    print(preds)

if __name__ == "__main__":
    logistic_regression_formula_example()
    print()
    poisson_regression_formula_example()
