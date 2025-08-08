# examples/predict_with_formula_example.py

import pandas as pd
from statisticapy.models.linear_model import LinearRegression
from statisticapy.models.generalized_linear_model import LogisticRegression, PoissonRegression

def linear_regression_predict_formula():
    data = pd.DataFrame({
        'y': [2, 4, 6, 8, 10],
        'x1': [1, 2, 3, 4, 5],
        'x2': [5, 4, 3, 2, 1]
    })

    formula = 'y ~ x1 + x2'
    model = LinearRegression(formula=formula)
    model.fit(data=data)

    new_data = pd.DataFrame({
        'x1': [6, 7],
        'x2': [0, -1]
    })

    preds = model.predict(new_data)
    print("Linear Regression Predictions with formula and DataFrame input:")
    print(preds)

def logistic_regression_predict_formula():
    data = pd.DataFrame({
        'y': [0, 1, 0, 1, 1],
        'x1': [0.5, 1.2, 1.8, 3.0, 2.9],
        'x2': [1.2, 3.1, 2.8, 0.5, 2.7]
    })

    formula = 'y ~ x1 + x2'
    model = LogisticRegression(formula=formula)
    model.fit(data=data)

    new_data = pd.DataFrame({
        'x1': [2, 1],
        'x2': [1, 0.5]
    })

    preds = model.predict(new_data)
    print("Logistic Regression Predicted Probabilities with formula and DataFrame input:")
    print(preds)

def poisson_regression_predict_formula():
    data = pd.DataFrame({
        'y': [1, 3, 7, 6, 8],
        'x1': [1, 2, 3, 4, 5],
        'x2': [0.5, 1.5, 3.5, 0.5, 2.0]
    })

    formula = 'y ~ x1 + x2'
    model = PoissonRegression(formula=formula)
    model.fit(data=data)

    new_data = pd.DataFrame({
        'x1': [6, 7],
        'x2': [1, 2]
    })

    preds = model.predict(new_data)
    print("Poisson Regression Predicted Means with formula and DataFrame input:")
    print(preds)

if __name__ == "__main__":
    linear_regression_predict_formula()
    print()
    logistic_regression_predict_formula()
    print()
    poisson_regression_predict_formula()
