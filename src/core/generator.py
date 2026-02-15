"""
Модуль генерации синтетических временных рядов для тестирования.
"""

import numpy as np
import pandas as pd


def generate_coupled_system(
    n_samples: int = 500,
    coupling_strength: float = 0.8,
    noise_level: float = 0.2,
) -> pd.DataFrame:
    """
    Генерирует систему из 4 переменных:
    - X: авторегрессионный процесс (источник).
    - Y: зависит от X (X -> Y) с лагом 1.
    - Z: независимый шум (random walk).
    - S: сезонный компонент (синус).
    """
    np.random.seed(42)

    e_x = np.random.normal(0, 1, n_samples)
    e_y = np.random.normal(0, 1, n_samples)
    e_z = np.random.normal(0, 1, n_samples)

    x = np.zeros(n_samples)
    y = np.zeros(n_samples)

    for t in range(1, n_samples):
        x[t] = 0.5 * x[t - 1] + noise_level * e_x[t]
        y[t] = 0.5 * y[t - 1] + coupling_strength * x[t - 1] + noise_level * e_y[t]

    z = np.cumsum(e_z * noise_level)

    t_idx = np.arange(n_samples)
    s = np.sin(2 * np.pi * t_idx / 50) + np.random.normal(0, 0.1, n_samples)

    df = pd.DataFrame(
        {
            "Source (X)": x,
            "Target (Y)": y,
            "Noise (Z)": z,
            "Season (S)": s,
        }
    )

    return df.iloc[50:].reset_index(drop=True)


def generate_random_walks(n_vars: int = 5, n_samples: int = 500) -> pd.DataFrame:
    """Генерирует N случайных блужданий (часто дают ложные корреляции)."""
    np.random.seed(None)
    data = {}
    for i in range(n_vars):
        data[f"RW_{i + 1}"] = np.cumsum(np.random.normal(0, 1, n_samples))
    return pd.DataFrame(data)
