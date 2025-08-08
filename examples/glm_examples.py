# examples/glm_examples.py

import numpy as np
from statisticapy.models.generalized_linear_model import LogisticRegression, PoissonRegression

def logistic_regression_example():
    # Binary classification data
    X = np.array([[0.5, 1.2],
                  [1.3, 3.1],
                  [1.8, 2.8],
                  [3.0, 0.5],
                  [2.9, 2.7]])
    y = np.array([0, 1, 1, 0, 1])

    model = LogisticRegression()
    model.fit(X, y)
    preds = model.predict(X)
    print("Logistic Regression Predicted Probabilities:")
    print(preds)

def poisson_regression_example():
    # Count data examples
    X = np.array([[1.0, 0.5],
                  [2.0, 1.5],
                  [3.0, 3.5],
                  [4.0, 0.5],
                  [5.0, 2.0]])
    y = np.array([1, 3, 7, 6, 8])

    model = PoissonRegression()
    model.fit(X, y)
    preds = model.predict(X)
    print("Poisson Regression Predicted Means:")
    print(preds)

if __name__ == "__main__":
    logistic_regression_example()
    print()
    poisson_regression_example()
