# examples/formula_parser_example.py

import pandas as pd
from statisticapy.utils.formula_parser import parse_formula

def example():
    data = pd.DataFrame({
        'y': [1, 3, 5, 7, 9],
        'x1': [0, 1, 2, 3, 4],
        'x2': [1, 2, 3, 4, 5]
    })
    
    formula = 'y ~ x1 + x2'
    y, X = parse_formula(formula, data)
    
    print("Response (y):")
    print(y)
    print("\nDesign matrix (X):")
    print(X)

if __name__ == "__main__":
    example()
