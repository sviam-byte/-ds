"""Statistical diagnostics helpers used by the analysis engine."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from hurst import compute_Hc
from scipy.fft import fft
from scipy.signal import find_peaks
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



def shannon_entropy(series: pd.Series, bins: int = 32) -> float:
    """Шенноновская энтропия распределения значений (гистограмма).

    Это *не* энтропия последовательности, а энтропия эмпирического распределения.
    Удобна как «базовая» характеристика разнообразия значений.
    """
    arr = _coerce_1d_numeric(series)
    if arr.size < 10:
        return np.nan
    bins = int(max(4, min(bins, arr.size // 2)))
    hist, _ = np.histogram(arr, bins=bins, density=False)
    p = hist.astype(np.float64)
    p = p / (p.sum() + 1e-12)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p))) if p.size else np.nan


def permutation_entropy(series: pd.Series, order: int = 3, delay: int = 1, normalize: bool = True) -> float:
    """Permutation entropy (Bandt-Pompe).

    Оценивает «сложность» через частоты порядковых паттернов.
    normalize=True -> значение в [0,1].
    """
    arr = _coerce_1d_numeric(series)
    order = int(max(2, order))
    delay = int(max(1, delay))
    n = int(arr.size)
    m = order
    if n < (m - 1) * delay + m + 1:
        return np.nan
    idx = np.arange(m) * delay
    patterns = []
    for t in range(0, n - idx[-1]):
        w = arr[t + idx]
        if not np.all(np.isfinite(w)):
            continue
        patterns.append(tuple(np.argsort(w, kind="mergesort")))
    if not patterns:
        return np.nan
    from collections import Counter

    cnt = Counter(patterns)
    p = np.asarray(list(cnt.values()), dtype=np.float64)
    p = p / (p.sum() + 1e-12)
    pe = -np.sum(p * np.log2(p + 1e-12))
    if not normalize:
        return float(pe)
    import math

    return float(pe / (math.log2(math.factorial(m)) + 1e-12))


def fft_peaks(series: pd.Series, fs: float = 1.0, top_k: int = 3, peak_height_ratio: float = 0.2) -> dict:
    """Возвращает доминирующие частоты/периоды по FFT.

    Выход:
      {
        'freqs': [..], 'amps':[...], 'periods':[...]
      }

    Частоты в Гц, если fs в Гц; иначе в «циклах на отсчёт».
    """
    arr = _coerce_1d_numeric(series)
    n = int(arr.size)
    if n < 8:
        return {"freqs": [], "amps": [], "periods": []}
    fs = float(fs) if fs and np.isfinite(fs) and fs > 0 else 1.0
    dt = 1.0 / fs
    freqs = np.fft.fftfreq(n, d=dt)
    yf = fft(arr - np.mean(arr))
    amp = np.abs(yf)
    mask = freqs > 0
    freqs = freqs[mask]
    amp = amp[mask]
    if freqs.size == 0:
        return {"freqs": [], "amps": [], "periods": []}
    height = float(np.max(amp) * float(peak_height_ratio)) if np.isfinite(np.max(amp)) else 0.0
    peaks, props = find_peaks(amp, height=height)
    if peaks.size == 0:
        return {"freqs": [], "amps": [], "periods": []}
    pk = peaks[np.argsort(amp[peaks])[::-1]][: int(max(1, top_k))]
    pk_freqs = freqs[pk]
    pk_amps = amp[pk]
    with np.errstate(divide="ignore", invalid="ignore"):
        periods = np.where(pk_freqs > 0, 1.0 / pk_freqs, np.nan)
    return {
        "freqs": [float(x) for x in pk_freqs],
        "amps": [float(x) for x in pk_amps],
        "periods": [float(x) for x in periods],
    }


def detect_seasonality(series: pd.Series, fs: float = 1.0, max_period: int | None = None) -> dict:
    """Грубая детекция сезонности.

    Делает два сигнала:
      1) Пик по ACF (устойчиво к фазе)
      2) Пик по FFT (быстро)

    Возвращает:
      {
        'acf_period': int|None,
        'acf_strength': float|None,
        'fft_period': float|None,
        'fft_freq': float|None,
      }
    """
    arr = _coerce_1d_numeric(series)
    n = int(arr.size)
    if n < 20:
        return {"acf_period": None, "acf_strength": None, "fft_period": None, "fft_freq": None}

    x = arr - np.mean(arr)
    x = x / (np.std(x) + 1e-12)
    acf = np.correlate(x, x, mode="full")[n - 1 :]
    acf = acf / (acf[0] + 1e-12)
    max_lag = int(min(n - 1, max_period if max_period is not None else max(5, n // 2)))
    lags = np.arange(1, max_lag + 1)
    acf_seg = acf[1 : max_lag + 1]
    pks, _ = find_peaks(acf_seg, height=0.1)
    if pks.size:
        best_idx = int(pks[np.argmax(acf_seg[pks])])
        acf_period = int(lags[best_idx])
        acf_strength = float(acf_seg[best_idx])
    else:
        acf_period, acf_strength = None, None

    pk = fft_peaks(series, fs=fs, top_k=1)
    if pk["freqs"]:
        fft_freq = float(pk["freqs"][0])
        fft_period = float(pk["periods"][0]) if np.isfinite(pk["periods"][0]) else None
    else:
        fft_freq, fft_period = None, None

    return {
        "acf_period": acf_period,
        "acf_strength": acf_strength,
        "fft_period": fft_period,
        "fft_freq": fft_freq,
    }
