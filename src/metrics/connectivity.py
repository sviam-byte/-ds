"""Метрики связности, выделенные из основного движка анализа."""

from __future__ import annotations

import importlib
import logging
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.spatial import cKDTree
from scipy.special import digamma
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR

from ..config import DEFAULT_BINS, DEFAULT_K_MI, DEFAULT_MAX_LAG, PYINFORM_AVAILABLE


def correlation_matrix(data: pd.DataFrame, lag: int = 1, control: Optional[list[str]] = None, **_: dict) -> np.ndarray:
    """Вычисляет матрицу корреляции Пирсона для числовых колонок."""
    return data.corr().values


def partial_correlation_matrix(df: pd.DataFrame, lag: int = 1, control: Optional[list[str]] = None, **_: dict) -> np.ndarray:
    """Вычисляет матрицу частных корреляций через матрицу точности."""
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    n_cols = len(cols)
    n_rows = len(df)

    # Защита от проклятия размерности: при N <= P инверсия/псевдоинверсия крайне нестабильна.
    if n_rows <= n_cols + 2:
        logging.warning(
            "Слишком мало данных для Partial Correlation: строк %s <= колонок %s. Возвращаю NaN.",
            n_rows,
            n_cols,
        )
        return np.full((n_cols, n_cols), np.nan)

    out = np.eye(n_cols)
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            xi, xj = cols[i], cols[j]
            ctrl_vars = control if control is not None else [c for c in cols if c not in (xi, xj)]
            sub_cols = [xi, xj] + [c for c in ctrl_vars if c in cols and c not in (xi, xj)]
            sub = df[sub_cols].dropna()
            if sub.shape[0] < len(sub_cols) + 1:
                pcor = np.nan
            else:
                try:
                    corr_matrix = sub.corr().values
                    precision = np.linalg.pinv(corr_matrix)
                    pcor = -precision[0, 1] / np.sqrt(precision[0, 0] * precision[1, 1])
                except Exception:
                    pcor = np.nan
            out[i, j] = out[j, i] = pcor
    return out


def partial_h2_matrix(df: pd.DataFrame, lag: int = 1, control: Optional[list[str]] = None, **kwargs: dict) -> np.ndarray:
    """Вычисляет квадрат частной корреляции (приближение частного H²)."""
    return partial_correlation_matrix(df, lag=lag, control=control, **kwargs) ** 2


def lagged_directed_correlation(df: pd.DataFrame, lag: int = 1, control: Optional[list[str]] = None, **_: dict) -> np.ndarray:
    """Вычисляет направленную лаговую корреляцию, где M[src, tgt] = corr(src(t), tgt(t+lag))."""
    lag = int(max(1, lag))
    cols = list(df.columns)
    n_cols = len(cols)
    out = np.full((n_cols, n_cols), np.nan, dtype=float)
    np.fill_diagonal(out, 0.0)

    for i, src in enumerate(cols):
        x_series = df[src]
        for j, tgt in enumerate(cols):
            if i == j:
                continue
            y_series = df[tgt].shift(-lag)
            pair = pd.concat([x_series, y_series], axis=1).dropna()
            if pair.shape[0] <= 3:
                continue
            xv = pair.iloc[:, 0].values
            yv = pair.iloc[:, 1].values
            if np.nanstd(xv) == 0 or np.nanstd(yv) == 0:
                out[i, j] = np.nan
            else:
                out[i, j] = float(np.corrcoef(xv, yv)[0, 1])
    return out


def _knn_mutual_info(x: np.ndarray, y: np.ndarray, k: int = DEFAULT_K_MI) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = int(min(x.size, y.size))
    if n <= k or n <= 3:
        return 0.0
    x = x[:n]
    y = y[:n]

    xy = np.c_[x, y]
    tree_xy = cKDTree(xy)
    distances, _ = tree_xy.query(xy, k=int(k) + 1, p=np.inf)
    eps = np.nextafter(distances[:, int(k)], 0.0)

    tree_x = cKDTree(x.reshape(-1, 1))
    tree_y = cKDTree(y.reshape(-1, 1))

    nx = np.fromiter(
        (max(0, len(tree_x.query_ball_point([x[i]], r=float(eps[i]), p=np.inf)) - 1) for i in range(n)),
        dtype=float,
        count=n,
    )
    ny = np.fromiter(
        (max(0, len(tree_y.query_ball_point([y[i]], r=float(eps[i]), p=np.inf)) - 1) for i in range(n)),
        dtype=float,
        count=n,
    )
    mi = digamma(n) + digamma(int(k)) - np.mean(digamma(nx + 1.0) + digamma(ny + 1.0))
    return float(max(0.0, mi)) if np.isfinite(mi) else float("nan")


def _knn_conditional_mutual_info(x: np.ndarray, y: np.ndarray, z: np.ndarray, k: int = DEFAULT_K_MI) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    n = int(min(x.size, y.size, z.shape[0]))
    if n <= k or n <= 3:
        return 0.0
    x = x[:n]
    y = y[:n]
    z = z[:n, :]

    xz = np.c_[x, z]
    yz = np.c_[y, z]
    xyz = np.c_[x, y, z]
    tree_xyz = cKDTree(xyz)
    distances, _ = tree_xyz.query(xyz, k=int(k) + 1, p=np.inf)
    eps = np.nextafter(distances[:, int(k)], 0.0)

    tree_xz = cKDTree(xz)
    tree_yz = cKDTree(yz)
    tree_z = cKDTree(z)

    nxz = np.fromiter((max(0, len(tree_xz.query_ball_point(xz[i], r=float(eps[i]), p=np.inf)) - 1) for i in range(n)), dtype=float, count=n)
    nyz = np.fromiter((max(0, len(tree_yz.query_ball_point(yz[i], r=float(eps[i]), p=np.inf)) - 1) for i in range(n)), dtype=float, count=n)
    nz = np.fromiter((max(0, len(tree_z.query_ball_point(z[i], r=float(eps[i]), p=np.inf)) - 1) for i in range(n)), dtype=float, count=n)

    cmi = digamma(int(k)) - np.mean(digamma(nxz + 1.0) + digamma(nyz + 1.0) - digamma(nz + 1.0))
    return float(max(0.0, cmi)) if np.isfinite(cmi) else float("nan")


def mutual_info_matrix(data: pd.DataFrame, lag: int = 1, control: Optional[list[str]] = None, k: int = DEFAULT_K_MI, **_: dict) -> np.ndarray:
    """Вычисляет попарную матрицу взаимной информации оценкой KSG kNN."""
    n_vars = len(data.columns)
    mi_matrix = np.zeros((n_vars, n_vars), dtype=float)
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            pair = data.iloc[:, [i, j]].dropna()
            if pair.shape[0] <= k:
                value = np.nan
            else:
                value = _knn_mutual_info(pair.iloc[:, 0].values, pair.iloc[:, 1].values, k=k)
            mi_matrix[i, j] = mi_matrix[j, i] = value
    return mi_matrix


def mutual_info_matrix_partial(data: pd.DataFrame, lag: int = 1, control: Optional[list[str]] = None, k: int = DEFAULT_K_MI, **_: dict) -> np.ndarray:
    """Вычисляет матрицу условной взаимной информации I(X;Y|Z)."""
    cols = list(data.columns)
    n_cols = len(cols)
    pmi = np.zeros((n_cols, n_cols), dtype=float)
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            xi, xj = cols[i], cols[j]
            z_cols = control if control is not None else [c for c in cols if c not in (xi, xj)]
            z_cols = [c for c in z_cols if c in data.columns and c not in (xi, xj)]
            if not z_cols:
                pair = data[[xi, xj]].dropna()
                value = _knn_mutual_info(pair[xi].values, pair[xj].values, k=k) if pair.shape[0] > k else np.nan
            else:
                sub = data[[xi, xj] + z_cols].dropna()
                value = _knn_conditional_mutual_info(sub[xi].values, sub[xj].values, sub[z_cols].values, k=k) if sub.shape[0] > k else np.nan
            pmi[i, j] = pmi[j, i] = value
    return pmi


def coherence_matrix(data: pd.DataFrame, lag: int = 1, control: Optional[list[str]] = None, fs: float = 1.0, **_: dict) -> np.ndarray:
    """Вычисляет среднюю спектральную когерентность для каждой пары переменных."""
    fs = fs if np.isfinite(fs) and fs > 0 else 1.0
    n_vars = data.shape[1]
    coh = np.full((n_vars, n_vars), np.nan, dtype=float)
    np.fill_diagonal(coh, 1.0)

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            pair = data.iloc[:, [i, j]].dropna()
            if pair.shape[0] <= 3:
                continue
            s1 = np.asarray(pair.iloc[:, 0].values, dtype=np.float64)
            s2 = np.asarray(pair.iloc[:, 1].values, dtype=np.float64)
            n = int(min(s1.size, s2.size))
            if n <= 3:
                continue
            nperseg = int(max(8, min(64, n // 2)))
            try:
                _, cxy = signal.coherence(s1[:n], s2[:n], fs=fs, nperseg=nperseg, detrend="constant")
                cxy = np.clip(np.asarray(cxy, dtype=np.float64), 0.0, 1.0)
                cxy[~np.isfinite(cxy)] = np.nan
                coh[i, j] = coh[j, i] = float(np.nanmean(cxy)) if np.isfinite(cxy).any() else np.nan
            except Exception:
                coh[i, j] = coh[j, i] = np.nan
    return coh


def granger_matrix(df: pd.DataFrame, lag: int = DEFAULT_MAX_LAG, control: Optional[list[str]] = None, **_: dict) -> np.ndarray:
    """Вычисляет матрицу p-value теста Грейнджера по попарным проверкам."""
    n_cols = df.shape[1]
    out = np.full((n_cols, n_cols), 1.0)
    columns = df.columns.tolist()
    for src in range(n_cols):
        for tgt in range(n_cols):
            if src == tgt:
                out[src, tgt] = 0.0
                continue
            pair = df[[columns[tgt], columns[src]]].dropna()
            if len(pair) <= lag * 2 + 5:
                out[src, tgt] = np.nan
                continue
            try:
                tests = grangercausalitytests(pair, maxlag=lag, verbose=False)
                out[src, tgt] = tests[lag][0]["ssr_ftest"][1]
            except Exception:
                out[src, tgt] = np.nan
    return out


def granger_matrix_partial(df: pd.DataFrame, lag: int = DEFAULT_MAX_LAG, control: Optional[list[str]] = None, **_: dict) -> np.ndarray:
    """Вычисляет условную причинность Грейнджера (p-value) через многомерную VAR."""
    columns = list(df.columns)
    n_cols = len(columns)
    out = np.full((n_cols, n_cols), np.nan, dtype=float)
    np.fill_diagonal(out, 0.0)

    # Для VAR-модели нужно достаточно наблюдений относительно числа признаков.
    if len(df) <= n_cols + 2:
        logging.warning(
            "Слишком мало данных для Granger partial: строк %s <= колонок %s. Возвращаю NaN.",
            len(df),
            n_cols,
        )
        return out

    for src_i, src in enumerate(columns):
        for tgt_j, tgt in enumerate(columns):
            if src_i == tgt_j:
                continue
            control_cols = control if control is not None else [c for c in columns if c not in (src, tgt)]
            control_cols = [c for c in control_cols if c in df.columns and c not in (src, tgt)]
            use_cols = [tgt, src] + control_cols
            sub = df[use_cols].dropna()
            p = int(max(1, lag))
            if sub.shape[0] < max(30, 5 * p * len(use_cols)):
                continue
            try:
                result = VAR(sub).fit(maxlags=p, ic=None, trend="c")
                causality = result.test_causality(caused=tgt, causing=[src], kind="f")
                out[src_i, tgt_j] = float(causality.pvalue) if np.isfinite(causality.pvalue) else np.nan
            except Exception:
                out[src_i, tgt_j] = np.nan
    return out


def _load_pyinform():
    if not PYINFORM_AVAILABLE:
        return None
    return importlib.import_module("pyinform")


def _transfer_entropy_discrete(source_d: np.ndarray, target_d: np.ndarray, k: int = 1) -> float:
    k = max(1, int(k))
    source_d = np.asarray(source_d, dtype=int).ravel()
    target_d = np.asarray(target_d, dtype=int).ravel()
    n = min(source_d.size, target_d.size)
    if n <= k + 1:
        return float("nan")
    source_d = source_d[:n]
    target_d = target_d[:n]

    c_xyz, c_xx, c_xpast_ypast, c_xpast = Counter(), Counter(), Counter(), Counter()
    for t in range(k, n):
        x_t = int(target_d[t])
        x_past = tuple(int(v) for v in target_d[t - k : t])
        y_past = tuple(int(v) for v in source_d[t - k : t])
        c_xyz[(x_t, x_past, y_past)] += 1
        c_xx[(x_t, x_past)] += 1
        c_xpast_ypast[(x_past, y_past)] += 1
        c_xpast[x_past] += 1

    n_eff = n - k
    te = 0.0
    for (x_t, x_past, y_past), count in c_xyz.items():
        p_xyz = count / n_eff
        p_x_given_x_y = count / c_xpast_ypast[(x_past, y_past)]
        p_x_given_x = c_xx[(x_t, x_past)] / c_xpast[x_past]
        if p_x_given_x_y > 0 and p_x_given_x > 0:
            te += p_xyz * np.log2(p_x_given_x_y / p_x_given_x)
    return float(te)


def compute_te_jitter(source: np.ndarray, target: np.ndarray, lag: int = 1, bins: int = DEFAULT_BINS) -> float:
    """Вычисляет Transfer Entropy с использованием z-score, jitter и квантильной дискретизации."""

    def _zscore_1d(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size == 0:
            return x
        mean = np.nanmean(x)
        std = np.nanstd(x)
        return x - mean if (not np.isfinite(std) or std <= 0) else (x - mean) / std

    def _add_tiny_jitter(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.size <= 3:
            return x
        uniq = np.unique(x[np.isfinite(x)])
        if uniq.size < max(3, int(0.2 * x.size)):
            rng = np.random.default_rng(0)
            scale = (np.nanstd(x) if np.nanstd(x) > 0 else 1.0) * 1e-10
            x = x + rng.normal(0.0, scale, size=x.shape)
        return x

    def discretize_quantile(series: np.ndarray, num_bins: int) -> np.ndarray:
        s = _add_tiny_jitter(_zscore_1d(series))
        s = s[np.isfinite(s)]
        if s.size == 0:
            return np.array([], dtype=int)
        if float(np.nanmin(s)) == float(np.nanmax(s)):
            return np.zeros(int(np.asarray(series).size), dtype=int)
        edges = np.unique(np.quantile(s, np.linspace(0.0, 1.0, int(num_bins) + 1)))
        if edges.size <= 2:
            return np.zeros(int(np.asarray(series).size), dtype=int)
        edges[-1] = np.nextafter(edges[-1], edges[-1] + 1.0)
        disc = np.digitize(_add_tiny_jitter(_zscore_1d(np.asarray(series))), bins=edges[1:-1], right=False)
        return np.clip(disc, 0, int(num_bins) - 1).astype(int)

    try:
        source_discrete = discretize_quantile(source, bins)
        target_discrete = discretize_quantile(target, bins)
        pyinform = _load_pyinform()
        k = int(max(1, lag))
        if pyinform is not None:
            return float(pyinform.transfer_entropy(source_discrete, target_discrete, k=k))
        return _transfer_entropy_discrete(source_discrete, target_discrete, k=k)
    except Exception as exc:
        logging.error("[TE] Ошибка вычисления: %s", exc)
        return float("nan")


def transfer_entropy_matrix(df: pd.DataFrame, lag: int = 1, control: Optional[list[str]] = None, bins: int = DEFAULT_BINS, **_: dict) -> np.ndarray:
    """Вычисляет направленную матрицу Transfer Entropy M[src, tgt] = TE(src -> tgt)."""
    n_cols = df.shape[1]
    out = np.zeros((n_cols, n_cols))
    for src in range(n_cols):
        for tgt in range(n_cols):
            if src == tgt:
                continue
            pair = df.iloc[:, [src, tgt]].dropna()
            if len(pair) <= lag:
                out[src, tgt] = np.nan
                continue
            out[src, tgt] = compute_te_jitter(pair.iloc[:, 0].values, pair.iloc[:, 1].values, lag=lag, bins=bins)
    return out


def transfer_entropy_matrix_partial(df: pd.DataFrame, lag: int = 1, control: Optional[list[str]] = None, bins: int = DEFAULT_BINS, **_: dict) -> np.ndarray:
    """Вычисляет приближенную частную Transfer Entropy через линейную резидуализацию."""
    cols = list(df.columns)
    n_cols = len(cols)
    out = np.zeros((n_cols, n_cols))

    def residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return y
        x_aug = np.c_[np.ones(len(x)), x]
        beta, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
        return y - x_aug @ beta

    for i, src in enumerate(cols):
        for j, tgt in enumerate(cols):
            if i == j:
                continue
            ctrl_cols = control if control is not None else [c for c in cols if c not in (src, tgt)]
            ctrl_cols = [c for c in ctrl_cols if c in df.columns]
            sub = df[[src, tgt] + ctrl_cols].dropna()
            if sub.shape[0] <= lag + 1:
                out[i, j] = np.nan
                continue
            x_ctrl = sub[ctrl_cols].values if ctrl_cols else np.empty((len(sub), 0))
            try:
                src_res = residualize(sub[src].values, x_ctrl)
                tgt_res = residualize(sub[tgt].values, x_ctrl)
                out[i, j] = compute_te_jitter(src_res, tgt_res, lag=lag, bins=bins)
            except Exception:
                out[i, j] = np.nan
    return out
