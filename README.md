# StatisticaPy

[![PyPI version](https://img.shields.io/pypi/v/statisticapy.svg)](https://pypi.org/project/statisticapy/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/yourusername/statisticapy/ci.yml)](https://github.com/yourusername/statisticapy/actions)
[![Coverage](https://img.shields.io/codecov/c/github/yourusername/statisticapy.svg)](https://codecov.io/gh/yourusername/statisticapy)

---

StatisticaPy is a next-generation Python library providing advanced, scalable, and user-friendly tools for **statistical modeling**, **hypothesis testing**, **time series analysis**, and **diagnostics** — specifically tailored for researchers, data scientists, and analysts across finance, economics, biostatistics, and social sciences.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Modules](#core-modules)
- [Usage Examples](#usage-examples)
- [Advanced Features & Roadmap](#advanced-features--roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Features

- **Comprehensive Statistical Modeling:**
  - Linear regression (OLS, GLS, WLS)
  - Generalized linear models (Logistic, Poisson, etc.) with various link functions
  - Time series models including ARIMA, state-space, Kalman filters
  - Multivariate models including MANOVA and GEE

- **Rich Diagnostics and Hypothesis Testing:**
  - Parametric and non-parametric tests (t-tests, ANOVA, chi-square, Wilcoxon, KS)
  - Robust statistics (skewness, kurtosis, outlier detection)
  - Residual analysis, heteroscedasticity, influence measures (leverage, Cook’s distance)

- **User-Friendly Interface:**
  - Support for R-style formula specification for model design
  - Seamless integration with pandas DataFrames and Series
  - Clear error messages and warnings to guide users

- **High Performance and Scalability:**
  - Vectorized computations leveraging NumPy and SciPy
  - Optional Cython acceleration planned for computational kernels
  - Supports parallel and distributed computations for intensive tasks

- **Visualization Utilities:**
  - Diagnostic plots such as residuals vs fitted, QQ plots, influence plots
  - High-level API built on Matplotlib and optional Seaborn support

- **Extensible and Modular Architecture:**
  - Modular package structure for easy extension and maintenance
  - Community-friendly with comprehensive tests, documentation, and contribution guidelines

---

## Installation

StatisticaPy requires Python 3.7+.

You can install the stable release via PyPI:

pip install statisticapy



Alternatively, clone the repository and install locally:

git clone https://github.com/yourusername/statisticapy.git
cd statisticapy
pip install -e .



**Dependencies include:** `numpy`, `scipy`, `pandas`, `matplotlib`, and (optionally) `seaborn`, `patsy`.

---

## Quick Start

Here is a minimal example demonstrating linear regression with formula interface:

import pandas as pd
from statisticapy.models.linear_model import LinearRegression

Prepare data
data = pd.DataFrame({
'y': [2, 4,
'x1': [1, 2,
'x2': [5, 4,Define and fit model using formula
model = LinearRegression(formula='y ~ x1 + x2')
model.fit(data=data)

Predict on new data
new_data = pd.DataFrame({'x1': [6,x2': [0, -1]})
predictions = model.predict(new_data)
print("Predictions:", predictions)



---

## Core Modules Overview

- `statisticapy.core`: Base classes and fitting engines
- `statisticapy.models`:
  - `linear_model.py`: Linear Regression (OLS, intercept handling, formula support)
  - `generalized_linear_model.py`: GLMs including Logistic and Poisson regression, IRLS estimation
  - `time_series.py`, `state_space.py`, `gee.py`, `multivariate.py` (coming soon)
- `statisticapy.diagnostics`:
  - `hypothesis_tests.py`: Parametric & non-parametric test implementations
  - `robust_stats.py`: Robust skewness, kurtosis, outlier detection
  - `residuals.py`: Residual computation and diagnostic tests
  - `influence.py`: Influence measures and Cook’s distance diagnostics
- `statisticapy.utils`:
  - `formula_parser.py`: R-style formula parsing with optional `patsy` backend
  - `visualization.py`: Diagnostic plotting utilities
  - `data_io.py`: Helpers for data import/export (CSV, Excel, SQL)
  - `parallel.py`: Parallel computing utilities (planned)
  - `math_helpers.py`: Numerical optimization and math routines

---

## Usage Examples

Visit the `/examples` directory for hands-on Jupyter Notebooks and example scripts covering:

- Linear Regression with and without formulas
- GLMs: Logistic and Poisson regressions
- Diagnostic plots and influence analysis
- Time Series modeling & forecasting
- Data import/export workflows

---

## Advanced Features & Roadmap

- AI-powered model selection and automated diagnostics reporting
- Bayesian modules and advanced MCMC sampling
- Support for big data, out-of-core and streaming
- Interactive dashboards with Plotly Dash or Streamlit
- Spatial statistics, survival analysis, and high-dimensional methods
- Cross-language interoperability with R and Julia

---

## Contributing

We welcome contributors! Please review the [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines on:

- How to report issues
- Coding style and best practices
- Writing tests and documentation
- Pull request process

Join the community discussions and help make StatisticaPy even better.

---

## License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.

---

## Contact

For questions, feedback, or support:

- GitHub Issues: https://github.com/yourusername/statisticapy/issues
- Email: your-email@example.com
- Twitter: [@yourhandle](https://twitter.com/yourhandle)

---

Thank you for choosing **StatisticaPy** for your statistical modeling needs!

---
