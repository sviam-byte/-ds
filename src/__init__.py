#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time Series Analysis Tool - главный пакет.
"""

from .config import *
from .core.data_loader import load_or_generate
from .core.preprocessing import configure_warnings

__all__ = [
    "load_or_generate",
    "configure_warnings",
    "PYINFORM_AVAILABLE",
    "DEFAULT_MAX_LAG",
    "DEFAULT_PVALUE_ALPHA",
    "DEFAULT_EDGE_THRESHOLD",
]
