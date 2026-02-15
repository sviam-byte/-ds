"""Unit tests for connectivity metrics registry and core behavior."""

import numpy as np
import pandas as pd

from src.metrics.connectivity import correlation_matrix, granger_matrix
from src.metrics.registry import get_metric_func


def test_correlation_identity() -> None:
    """Correlation of identical columns should be 1 for all entries."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    matrix = correlation_matrix(df)
    assert np.allclose(matrix, [[1.0, 1.0], [1.0, 1.0]], equal_nan=True)


def test_correlation_nan_handling() -> None:
    """Pairwise NaN handling should still return finite perfect correlation for aligned points."""
    df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [1, 2, 3, 4]})
    matrix = correlation_matrix(df)
    assert np.isfinite(matrix[0, 1])
    assert matrix[0, 1] > 0.99


def test_granger_empty_safe() -> None:
    """Granger matrix should not crash on too-short inputs and should return NxN shape."""
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, 2.0]})
    matrix = granger_matrix(df, lag=5)
    assert matrix.shape == (2, 2)


def test_registry_lookup() -> None:
    """Registry should return callable metric by canonical name."""
    func = get_metric_func("correlation_full")
    out = func(pd.DataFrame({"x": [0, 1], "y": [0, 1]}), lag=1, control=None)
    assert out.shape == (2, 2)
