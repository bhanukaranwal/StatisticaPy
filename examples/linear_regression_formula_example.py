# examples/linear_regression_formula_example.py

import pandas as pd
from statisticapy.models.linear_model import LinearRegression

def example():
    data = pd.DataFrame({
        'y': [1, 3, 5, 7, 9],
        'x1': [0, 1, 2, 3, 4],
        'x2': [1, 2, 3, 4, 5]
    })
    formula = 'y ~ x1 + x2'
    
    model = LinearRegression(formula=formula)
    model.fit(data=data)
    print("Fitted parameters:")
    print(model.params_)
    
    # Prediction with numeric array (must include predictors, intercept auto-handled internally)
    X_new = [[5, 6], [7, 8]]
    preds = model.predict(X_new)
    print("Predictions for new data:")
    print(preds)

if __name__ == "__main__":
    example()
