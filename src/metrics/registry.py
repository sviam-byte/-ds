"""Central metrics registry used by the engine orchestrator."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import connectivity

MetricFunc = Callable[[pd.DataFrame, int, Optional[list[str]]], np.ndarray]


def _h2_full(df: pd.DataFrame, lag: int = 1, control: Optional[list[str]] = None, **kwargs: dict) -> np.ndarray:
    return connectivity.correlation_matrix(df, lag=lag, control=control, **kwargs) ** 2


def _h2_directed(df: pd.DataFrame, lag: int = 1, control: Optional[list[str]] = None, **kwargs: dict) -> np.ndarray:
    return connectivity.lagged_directed_correlation(df, lag=lag, control=control, **kwargs) ** 2


METRICS_REGISTRY: dict[str, MetricFunc] = {
    "correlation_full": connectivity.correlation_matrix,
    "correlation_partial": connectivity.partial_correlation_matrix,
    "correlation_directed": connectivity.lagged_directed_correlation,
    "h2_full": _h2_full,
    "h2_partial": connectivity.partial_h2_matrix,
    "h2_directed": _h2_directed,
    "mutinf_full": connectivity.mutual_info_matrix,
    "mutinf_partial": connectivity.mutual_info_matrix_partial,
    "coherence_full": connectivity.coherence_matrix,
    "granger_full": connectivity.granger_matrix,
    "granger_partial": connectivity.granger_matrix_partial,
    "granger_directed": connectivity.granger_matrix,
    "te_full": connectivity.transfer_entropy_matrix,
    "te_partial": connectivity.transfer_entropy_matrix_partial,
    "te_directed": connectivity.transfer_entropy_matrix,
}


def register_metric(name: str, func: MetricFunc) -> None:
    """Register or override metric function in the central registry."""
    METRICS_REGISTRY[name] = func


def get_metric_func(name: str) -> MetricFunc:
    """Return metric function by name, raising ValueError for unknown metrics."""
    if name not in METRICS_REGISTRY:
        raise ValueError(f"Unknown metric: {name}")
    return METRICS_REGISTRY[name]
