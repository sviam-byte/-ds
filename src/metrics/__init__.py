"""Metrics package exports connectivity functions and registry helpers."""

from . import connectivity
from .registry import METRICS_REGISTRY, get_metric_func, register_metric

__all__ = ["connectivity", "METRICS_REGISTRY", "get_metric_func", "register_metric"]
