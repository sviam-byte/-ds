#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Модуль предобработки временных рядов.
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats


def additional_preprocessing(df: pd.DataFrame, unique_thresh: float = 0.05) -> pd.DataFrame:
    """
    Дополнительная предобработка данных:
    - Удаление почти константных колонок
    - Лог-преобразование для снижения асимметрии
    
    Args:
        df: Исходный DataFrame
        unique_thresh: Порог уникальности для удаления константных колонок
        
    Returns:
        pd.DataFrame: Предобработанный DataFrame
    """
    df = df.copy()
    
    # Удаляем почти константные колонки
    for col in df.columns:
        if len(df[col]) > 0 and pd.api.types.is_numeric_dtype(df[col]):
            uniq_ratio = df[col].nunique() / len(df[col])
            if uniq_ratio < unique_thresh:
                logging.info(
                    f"[Preproc] Столбец {col} почти константный (uniq_ratio={uniq_ratio:.3f}), удаляем."
                )
                df.drop(columns=[col], inplace=True)
    
    # Лог-преобразование для снижения асимметрии
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and (df[col] > 0).all():
            skew_before = stats.skew(df[col].dropna())
            if not np.isnan(skew_before):
                transformed = np.log(df[col])
                skew_after = stats.skew(transformed.dropna())
                if not np.isnan(skew_after) and abs(skew_after) < abs(skew_before):
                    logging.info(
                        f"[Preproc] Лог-преобразование для {col}: skew {skew_before:.3f} -> {skew_after:.3f}."
                    )
                    df[col] = transformed
    
    return df


def configure_warnings(quiet: bool = False) -> None:
    """
    Настраивает предупреждения без глобального подавления.
    
    Args:
        quiet: Если True, подавляет все предупреждения
    """
    import warnings
    
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module="statsmodels.tsa.stattools",
    )
    warnings.filterwarnings(
        "ignore",
        message="nperseg = 256 is greater than input length",
    )
    if quiet:
        warnings.filterwarnings("ignore")
