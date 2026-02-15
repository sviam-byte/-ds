"""Statistical diagnostics helpers used by the analysis engine."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from hurst import compute_Hc
from scipy.fft import fft
from statsmodels.tsa.stattools import adfuller

try:
    import nolds
except ImportError:  # pragma: no cover - optional dependency path
    nolds = None


def _coerce_1d_numeric(series_like) -> np.ndarray:
    """Convert input to a finite 1D float64 array."""
    try:
        s = pd.to_numeric(series_like, errors="coerce")
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        if isinstance(s, (pd.Series, pd.Index)):
            arr = s.to_numpy()
        else:
            arr = np.asarray(s)
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
        return arr[np.isfinite(arr)]
    except Exception:
        arr = np.asarray(series_like, dtype=np.float64).reshape(-1)
        return arr[np.isfinite(arr)]


def test_stationarity(series: pd.Series) -> tuple[float | None, float | None]:
    """Run the Augmented Dickey-Fuller test and return (statistic, p-value)."""
    clean_series = series.dropna()
    if len(clean_series) < 5:
        return None, None
    try:
        res = adfuller(clean_series)
        return float(res[0]), float(res[1])
    except Exception as exc:
        logging.warning("ADF error: %s", exc)
        return None, None


def compute_hurst_rs(series: pd.Series) -> float:
    """Calculate the Hurst exponent using R/S analysis."""
    try:
        arr = _coerce_1d_numeric(series)
        if arr.size < 20:
            return np.nan
        try:
            hurst_exp, _, _ = compute_Hc(arr, kind="change", simplified=True)
            return float(hurst_exp)
        except Exception:
            if nolds is None:
                return np.nan
            return float(nolds.hurst_rs(arr))
    except Exception as exc:
        logging.error("[Hurst RS] %s", exc)
        return np.nan


def compute_hurst_dfa(series: pd.Series) -> float:
    """Calculate the Hurst exponent using detrended fluctuation analysis."""
    try:
        arr = _coerce_1d_numeric(series)
        if arr.size < 20 or nolds is None:
            return np.nan
        return float(nolds.dfa(arr))
    except Exception as exc:
        logging.error("[Hurst DFA] %s", exc)
        return np.nan


def compute_hurst_aggvar(series: pd.Series, max_n: int = 100) -> float:
    """Calculate the Hurst exponent via the aggregated variance method."""
    try:
        arr = _coerce_1d_numeric(series)
        n = int(arr.size)
        if n < 50:
            return np.nan
        max_n = max(10, min(int(max_n), n // 2))
        variances, used_m = [], []
        for m in range(1, max_n + 1):
            nb = n // m
            if nb <= 1:
                continue
            block_means = arr[: nb * m].reshape(nb, m).mean(axis=1)
            if block_means.size <= 1:
                continue
            variance = np.var(block_means)
            if np.isfinite(variance) and variance > 0:
                variances.append(variance)
                used_m.append(m)
        if len(variances) < 2:
            return np.nan
        slope, _ = np.polyfit(np.log10(used_m), np.log10(variances), 1)
        return float(1.0 - slope / 2.0)
    except Exception as exc:
        logging.error("[Hurst AggVar] %s", exc)
        return np.nan


def compute_hurst_wavelet(series: pd.Series) -> float:
    """Estimate a wavelet-like Hurst proxy from log-log PSD slope."""
    try:
        arr = _coerce_1d_numeric(series)
        n = int(arr.size)
        if n < 50:
            return np.nan
        arr = arr - np.mean(arr)
        yf = fft(arr)
        freqs = np.fft.fftfreq(n)
        psd = np.abs(yf) ** 2
        mask = (freqs > 0) & (psd > 0) & np.isfinite(psd)
        freqs = freqs[mask]
        psd = psd[mask]
        if freqs.size < 2:
            return np.nan
        slope, _ = np.polyfit(np.log10(freqs), np.log10(psd), 1)
        return float((1.0 - slope) / 2.0)
    except Exception as exc:
        logging.error("[Hurst Wavelet] %s", exc)
        return np.nan


def compute_sample_entropy(series: pd.Series) -> float:
    """Compute sample entropy for a 1D series."""
    try:
        arr = _coerce_1d_numeric(series)
        if arr.size < 20 or np.std(arr) < 1e-10 or nolds is None:
            return np.nan
        return float(nolds.sampen(arr))
    except Exception as exc:
        logging.error("[Sample Entropy] %s", exc)
        return np.nan
