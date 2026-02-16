#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Модуль загрузки и парсинга данных из файлов.
"""

import logging
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config import DEFAULT_OUTLIER_Z
from .preprocessing import additional_preprocessing
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
import numpy as np


@dataclass
class PreprocessReport:
    """Структурированный отчёт о предобработке временных рядов.

    Используется в UI/HTML-отчёте, чтобы явно показать применённые шаги.
    """

    enabled: bool = True
    steps_global: List[str] = field(default_factory=list)
    steps_by_column: Dict[str, List[str]] = field(default_factory=dict)
    dropped_columns: List[str] = field(default_factory=list)
    notes: Dict[str, Any] = field(default_factory=dict)

    def add(self, msg: str, col: Optional[str] = None) -> None:
        """Добавляет шаг в глобальный список или к конкретной колонке."""
        if col is None:
            self.steps_global.append(msg)
        else:
            self.steps_by_column.setdefault(col, []).append(msg)


def _is_mostly_numeric_row(row) -> bool:
    """Проверяет, что в строке >=80% непустых значений приводятся к float."""
    vals = []
    for v in row:
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        vals.append(v)
    if not vals:
        return False
    numeric = 0
    for v in vals:
        try:
            float(v)
            numeric += 1
        except Exception:
            pass
    return numeric / max(1, len(vals)) >= 0.8


def _detect_header(df_raw: pd.DataFrame) -> bool:
    """Если 1-я строка нечисловая, а 2-я числовая — считаем 1-ю заголовком."""
    if df_raw.shape[0] < 2:
        return False
    r0 = df_raw.iloc[0].tolist()
    r1 = df_raw.iloc[1].tolist()
    return (not _is_mostly_numeric_row(r0)) and _is_mostly_numeric_row(r1)


def _maybe_split_single_column(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Поддержка формата: одна колонка строк, внутри ',' ';' '\\t'."""
    if df_raw.shape[1] == 1 and isinstance(df_raw.iloc[0, 0], str):
        return df_raw[0].astype(str).str.split(r"[,;\t]", expand=True)
    return df_raw


def _detect_time_like_col(col: pd.Series) -> bool:
    """Эвристика для авто-обнаружения временной/индексной колонки."""
    try:
        dt = pd.to_datetime(col, errors="coerce", utc=False)
        if dt.notna().mean() >= 0.9:
            return dt.is_monotonic_increasing or dt.is_monotonic_decreasing
    except Exception:
        pass

    c = pd.to_numeric(col, errors="coerce")
    if c.notna().mean() >= 0.95:
        dif = c.dropna().diff().dropna()
        if len(dif) >= 3 and (dif.abs() > 0).mean() >= 0.9:
            return True
    return False


def read_input_table(filepath: str, header: str = "auto") -> pd.DataFrame:
    """Чтение CSV/XLSX с поддержкой автодетекта заголовка и одной строковой колонки."""
    fp = str(filepath)
    if fp.lower().endswith(".csv"):
        df0 = pd.read_csv(fp, header=None)
    else:
        df0 = pd.read_excel(fp, header=None)
    df0 = _maybe_split_single_column(df0)

    if header not in {"auto", "yes", "no"}:
        raise ValueError("header must be one of: auto|yes|no")
    has_header = _detect_header(df0) if header == "auto" else (header == "yes")
    if has_header:
        hdr = df0.iloc[0].astype(str).tolist()
        df = df0.iloc[1:].copy()
        df.columns = [h if h.strip() else f"c{i+1}" for i, h in enumerate(hdr)]
    else:
        df = df0.copy()
        df.columns = [f"c{i+1}" for i in range(df.shape[1])]
    return df


def tidy_timeseries_table(
    df: pd.DataFrame,
    time_col: str = "auto",
    transpose: str = "auto",
) -> pd.DataFrame:
    """Превращает сырую таблицу в numeric матрицу вида time × features."""
    out = df.copy()
    out = out.dropna(axis=1, how="all")

    if time_col not in {"auto", "none"} and time_col not in out.columns:
        raise ValueError(f"time_col '{time_col}' not found in columns")
    if time_col == "auto":
        if out.shape[1] >= 2 and _detect_time_like_col(out.iloc[:, 0]):
            out = out.iloc[:, 1:].copy()
    elif time_col != "none":
        out = out.drop(columns=[time_col])

    out = out.apply(pd.to_numeric, errors="coerce")
    good = [c for c in out.columns if out[c].notna().mean() >= 0.2]
    out = out[good]

    if transpose not in {"auto", "yes", "no"}:
        raise ValueError("transpose must be one of: auto|yes|no")
    do_t = (out.shape[0] < out.shape[1]) if transpose == "auto" else (transpose == "yes")
    if do_t:
        out = out.T
        out.columns = [f"c{i+1}" for i in range(out.shape[1])]

    out = out.dropna(axis=0, how="all")
    return out


def preprocess_timeseries(
    df: pd.DataFrame,
    *,
    enabled: bool = True,
    log_transform: bool = False,
    remove_outliers: bool = True,
    normalize: bool = True,
    fill_missing: bool = True,
    check_stationarity: bool = False,
    return_report: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, PreprocessReport]:
    """Предобработка матрицы (можно полностью отключить enabled=False)."""
    out = df.copy()
    report = PreprocessReport(enabled=bool(enabled))
    if not enabled:
        report.add("[Preprocess] disabled: using raw numeric matrix as-is.")
        return (out, report) if return_report else out

    report.add("[Preprocess] enabled")

    before_cols = list(out.columns)
    out = additional_preprocessing(out)
    after_cols = list(out.columns)
    dropped = [c for c in before_cols if c not in after_cols]
    if dropped:
        report.dropped_columns.extend(dropped)
        report.add(f"[Preprocess] dropped near-constant columns: {dropped}")

    out = out.fillna(out.mean(numeric_only=True))
    report.add("[Preprocess] fillna: column means")

    if log_transform:
        report.add("[Preprocess] log-transform: applied to positive values")
        out = out.applymap(lambda x: np.log(x) if x is not None and not np.isnan(x) and x > 0 else x)

    if remove_outliers:
        for col in out.columns:
            if pd.api.types.is_numeric_dtype(out[col]):
                series = out[col]
                mean, std = series.mean(skipna=True), series.std(skipna=True)
                if std > 0:
                    upper, lower = mean + DEFAULT_OUTLIER_Z * std, mean - DEFAULT_OUTLIER_Z * std
                    outliers = (series < lower) | (series > upper)
                    if outliers.any():
                        report.add(f"[Preprocess] outliers->NaN: z>{DEFAULT_OUTLIER_Z} (n={int(outliers.sum())})", col=col)
                        out.loc[outliers, col] = np.nan

    if fill_missing:
        report.add("[Preprocess] fill_missing: linear interpolate + bfill/ffill")
        out = out.interpolate(method="linear", limit_direction="both", axis=0).bfill().ffill().fillna(0)

    if normalize:
        report.add("[Preprocess] normalize: z-score")
        cols_to_norm = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
        if cols_to_norm:
            scaler = StandardScaler()
            out[cols_to_norm] = scaler.fit_transform(out[cols_to_norm])

    if check_stationarity:
        report.add("[Preprocess] stationarity check: ADF")
        for col in out.columns:
            if pd.api.types.is_numeric_dtype(out[col]):
                series = out[col].dropna()
                if len(series) > 10:
                    pvalue = adfuller(series, autolag="AIC")[1]
                    logging.info(
                        f"Ряд '{col}' {'стационарен' if pvalue <= 0.05 else 'вероятно нестационарен'} (p-value ADF={pvalue:.3f})."
                    )
    return (out, report) if return_report else out


def load_or_generate(
    filepath: str,
    *,
    header: str = "auto",
    time_col: str = "auto",
    transpose: str = "auto",
    preprocess: bool = True,
    log_transform: bool = False,
    remove_outliers: bool = True,
    normalize: bool = True,
    fill_missing: bool = True,
    check_stationarity: bool = False,
    return_report: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, PreprocessReport]:
    """
    Главная функция загрузки и предобработки данных из файла.
    
    Args:
        filepath: Путь к CSV или Excel файлу
        header: Режим заголовка ('auto', 'yes', 'no')
        time_col: Колонка времени ('auto', 'none', или название)
        transpose: Транспонирование ('auto', 'yes', 'no')
        preprocess: Включить предобработку
        log_transform: Применить логарифм
        remove_outliers: Удалить выбросы
        normalize: Нормализовать данные
        fill_missing: Заполнить пропуски
        check_stationarity: Проверить стационарность
        
    Returns:
        Если ``return_report=False``: ``pd.DataFrame`` с матрицей временных рядов.
        Если ``return_report=True``: кортеж ``(pd.DataFrame, PreprocessReport)``
        для последующей визуализации шагов предобработки в UI/HTML-отчёте.
    """
    try:
        raw = read_input_table(filepath, header=header)
        df = tidy_timeseries_table(raw, time_col=time_col, transpose=transpose)
        df_out = preprocess_timeseries(
            df,
            enabled=preprocess,
            log_transform=log_transform,
            remove_outliers=remove_outliers,
            normalize=normalize,
            fill_missing=fill_missing,
            check_stationarity=check_stationarity,
            return_report=bool(return_report),
        )
        if return_report:
            df, report = df_out  # type: ignore[misc]
        else:
            df, report = df_out, None
        logging.info(
            f"[Load] OK shape={df.shape} header={header} time_col={time_col} transpose={transpose} preprocess={preprocess}"
        )
        return (df, report) if return_report else df
    except Exception as e:
        logging.error(f"[Load] Ошибка загрузки: {e}")
        raise
