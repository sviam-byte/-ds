#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Конфигурация и константы для Time Series Analysis Tool.
"""

import importlib.util
import os
from dataclasses import dataclass

# Базовые пути
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_FOLDER = os.path.join(BASE_PATH, "TimeSeriesAnalysis")
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Параметры по умолчанию
DEFAULT_MAX_LAG = 5
DEFAULT_K_MI = 5
DEFAULT_BINS = 8
DEFAULT_OUTLIER_Z = 5
DEFAULT_REGULARIZATION = 1e-8
DEFAULT_EMBED_DIM = 3
DEFAULT_EMBED_TAU = 1
DEFAULT_PVALUE_ALPHA = 0.05
DEFAULT_EDGE_THRESHOLD = 0.2

# Параметр регуляризации
REG_ALPHA = 1e-5

# Проверка наличия pyinform
PYINFORM_AVAILABLE = importlib.util.find_spec("pyinform") is not None

# Стабильные методы
STABLE_METHODS = [
    "correlation_full",
    "correlation_partial",
    "coherence_full",
    "granger_full",
]

# Экспериментальные методы (базовый список)
EXPERIMENTAL_METHODS_BASE = [
    "mutinf_full",
    "mutinf_partial",
    "te_full",
    "te_partial",
    "te_directed",
    "ah_full",
    "ah_partial",
    "ah_directed",
]

# Если pyinform недоступен, скрываем TE-методы
EXPERIMENTAL_METHODS = [
    method
    for method in EXPERIMENTAL_METHODS_BASE
    if PYINFORM_AVAILABLE or not method.startswith("te_")
]

# P-value методы
PVAL_METHODS = {
    "granger_full",
    "granger_partial",
    "granger_directed",
}

# Directed методы
DIRECTED_METHODS = {
    "correlation_directed",
    "h2_directed",
    "granger_full",
    "granger_partial",
    "granger_directed",
    "te_full",
    "te_partial",
    "te_directed",
    "ah_full",
    "ah_partial",
    "ah_directed",
}

# Информация о методах
METHOD_INFO = {
    "correlation_full": {
        "title": "Корреляция (полная)",
        "meaning": "Линейная связь. Значение в [-1, 1]. |value| ближе к 1 = сильнее.",
    },
    "correlation_partial": {
        "title": "Частичная корреляция",
        "meaning": "Линейная связь при контроле остальных переменных. [-1, 1].",
    },
    "correlation_directed": {
        "title": "Лаговая корреляция (directed)",
        "meaning": "Оценка направленной связи через сдвиг по лагу. Чем больше |value|, тем сильнее.",
    },
    "mutinf_full": {
        "title": "Взаимная информация (MI)",
        "meaning": "Нелинейная зависимость. >= 0. Больше = сильнее.",
    },
    "mutinf_partial": {
        "title": "Частичная MI",
        "meaning": "MI при контроле переменных. >= 0. Больше = сильнее.",
    },
    "coherence_full": {
        "title": "Когерентность",
        "meaning": "Частотная синхронизация. Обычно в [0, 1]. Больше = сильнее.",
    },
    "h2_full": {
        "title": "H2 (полная)", 
        "meaning": "Нелинейная связность. Обычно в [0, 1]. Больше = сильнее."
    },
    "h2_partial": {
        "title": "H2 (partial)", 
        "meaning": "H2 при контроле. Обычно в [0, 1]. Больше = сильнее."
    },
    "h2_directed": {
        "title": "H2 (directed)", 
        "meaning": "Направленная H2. Больше = сильнее."
    },
    "granger_full": {
        "title": "Granger (p-values)", 
        "meaning": "p-value теста. Меньше = сильнее свидетельство причинности."
    },
    "granger_partial": {
        "title": "Granger partial (p-values)",
        "meaning": "Granger partial (linear control; best lag up to L): p-value после удаления влияния control. Меньше = сильнее.",
    },
    "granger_directed": {
        "title": "Granger directed (p-values)",
        "meaning": "То же семейство p-values. Меньше = сильнее.",
    },
    "te_full": {
        "title": "Transfer Entropy",
        "meaning": "Направленный поток информации. Больше = сильнее.",
    },
    "te_partial": {
        "title": "Transfer Entropy (partial)",
        "meaning": "TE при контроле. Больше = сильнее.",
    },
    "te_directed": {
        "title": "Transfer Entropy (directed)",
        "meaning": "TE (directed). Больше = сильнее.",
    },
    "ah_full": {
        "title": "Active information storage (AH)",
        "meaning": "Информационная мера памяти системы. Больше = сильнее.",
    },
    "ah_partial": {
        "title": "AH (partial)",
        "meaning": "AH при контроле. Больше = сильнее.",
    },
    "ah_directed": {
        "title": "AH (directed)",
        "meaning": "AH направленная. Больше = сильнее.",
    },
}




@dataclass(slots=True)
class AnalysisConfig:
    """Контейнер конфигурации анализа."""

    max_lag: int = DEFAULT_MAX_LAG
    p_value_alpha: float = DEFAULT_PVALUE_ALPHA
    graph_threshold: float = DEFAULT_EDGE_THRESHOLD
    enable_experimental: bool = False
    # Автоматически делать diff() если ряд нестационарен.
    auto_difference: bool = False
    # Поправка на множественные сравнения: 'none' | 'fdr_bh'.
    pvalue_correction: str = "none"

    # --- Connectivity tuning ---
    # Если задано, расчёты могут выполняться на скользящих окнах.
    # В результирующую матрицу по методу попадёт "лучшая" матрица (см. window_policy).
    window_sizes: list[int] | None = None
    # Шаг окна (в точках). Если None — дефолт = max(1, window_size//5).
    window_stride: int | None = None
    # Политика агрегации по окнам: 'best' | 'mean'.
    window_policy: str = "best"

    # Подбор лага: 'fixed' | 'optimize'
    # - fixed: используем lag=1 (или явно заданный)
    # - optimize: перебираем 1..max_lag и выбираем лучший по quality-score
    lag_selection: str = "optimize"

def is_pvalue_method(variant: str) -> bool:
    """Проверяет, является ли метод p-value методом."""
    return variant.lower() in PVAL_METHODS


def is_directed_method(variant: str) -> bool:
    """Проверяет, является ли метод направленным."""
    return variant.lower() in DIRECTED_METHODS


def is_control_sensitive_method(variant: str) -> bool:
    """Проверяет, зависит ли метод от control-переменных."""
    return "_partial" in variant.lower()
