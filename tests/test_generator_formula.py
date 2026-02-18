"""Тесты безопасного формульного генератора временных рядов."""

import numpy as np
import pytest

from src.core.generator import FormulaSpec, UnsafeFormulaError, generate_formula_dataset, safe_eval_vector


def test_generate_formula_dataset_is_reproducible_with_seed() -> None:
    """При одинаковом seed генератор должен быть детерминированным."""
    specs = [
        FormulaSpec("X", "sin(2*pi*t/50) + 0.2*randn()"),
        FormulaSpec("Y", "0.8*X + 0.3*randn()"),
    ]

    df1 = generate_formula_dataset(n_samples=64, seed=123, specs=specs)
    df2 = generate_formula_dataset(n_samples=64, seed=123, specs=specs)

    assert list(df1.columns) == ["X", "Y"]
    assert np.allclose(df1.values, df2.values)


def test_safe_eval_blocks_double_star_kwargs() -> None:
    """Распаковка **kwargs должна быть запрещена в sandbox-вычислении."""
    env = {"randn": lambda scale=1.0: np.zeros(5), "pi": np.pi}
    names = {"t": np.arange(5, dtype=float), "payload": {"scale": 1.0}}

    with pytest.raises(UnsafeFormulaError):
        safe_eval_vector("randn(**payload)", env=env, names=names)


def test_safe_eval_requires_1d_vector_output() -> None:
    """Формула не должна возвращать матрицу вместо одномерного ряда."""
    env = {"pi": np.pi}
    names = {"t": np.arange(4, dtype=float), "X": np.eye(4)}

    with pytest.raises(ValueError, match="1D-массив"):
        safe_eval_vector("X", env=env, names=names)
