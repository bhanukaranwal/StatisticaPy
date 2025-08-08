# statisticapy/__init__.py

from .core import fit_model
from .models import linear_model, generalized_linear_model, time_series, state_space, gee, multivariate
from .diagnostics import hypothesis_tests, robust_stats, residuals, influence
from .utils import formula_parser, data_io, visualization, parallel, math_helpers
from .ml_integration import ml_bridge

__version__ = "0.1.0"

__all__ = [
    "fit_model",
    "linear_model",
    "generalized_linear_model",
    "time_series",
    "state_space",
    "gee",
    "multivariate",
    "hypothesis_tests",
    "robust_stats",
    "residuals",
    "influence",
    "formula_parser",
    "data_io",
    "visualization",
    "parallel",
    "math_helpers",
    "ml_bridge",
]
