"""Модуль генерации синтетических временных рядов для тестирования.

Здесь есть два слоя:
1) готовые пресеты (coupled_system, random_walks)
2) генератор по формулам (x(t), y(t,x), z(t,x,y), ...)

Формулы вычисляются через безопасный парсер (AST) с белым списком функций.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import pandas as pd


def generate_coupled_system(
    n_samples: int = 500,
    coupling_strength: float = 0.8,
    noise_level: float = 0.2,
) -> pd.DataFrame:
    """Генерирует систему из 4 переменных:

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


# =========================
# Генерация по формулам
# =========================


@dataclass(frozen=True)
class FormulaSpec:
    """Описание одного ряда."""

    name: str
    expr: str


class UnsafeFormulaError(ValueError):
    """Формула содержит запрещённые конструкции."""


_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


def _validate_ast(node: ast.AST, allowed_names: set[str], allowed_funcs: set[str]) -> None:
    """Белый список AST-узлов: только арифметика и вызовы разрешённых функций."""

    for n in ast.walk(node):
        if isinstance(n, ast.Expression):
            continue
        if isinstance(n, ast.BinOp):
            if not isinstance(n.op, _ALLOWED_BINOPS):
                raise UnsafeFormulaError(f"Запрещённый оператор: {type(n.op).__name__}")
            continue
        if isinstance(n, ast.UnaryOp):
            if not isinstance(n.op, _ALLOWED_UNARYOPS):
                raise UnsafeFormulaError(f"Запрещённый унарный оператор: {type(n.op).__name__}")
            continue
        if isinstance(n, ast.Call):
            # Разрешаем только func(...) где func — имя из белого списка.
            if isinstance(n.func, ast.Name):
                fn = n.func.id
                if fn not in allowed_funcs:
                    raise UnsafeFormulaError(f"Запрещённая функция: {fn}")
            else:
                raise UnsafeFormulaError("Запрещены атрибуты/лямбды/индексации в вызовах")
            continue
        if isinstance(n, ast.Name):
            if n.id not in allowed_names and n.id not in allowed_funcs:
                raise UnsafeFormulaError(f"Запрещённое имя: {n.id}")
            continue
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)) or n.value is None:
                continue
            raise UnsafeFormulaError("Разрешены только числовые константы")
        if isinstance(n, ast.Tuple):
            # полезно для where(cond, a, b) не нужно; но оставляем для совместимости
            continue
        if isinstance(n, ast.keyword):
            continue

        # Явно запрещаем всё остальное: Attribute, Subscript, Compare, BoolOp, IfExp, Comprehension, etc.
        if isinstance(
            n,
            (
                ast.Attribute,
                ast.Subscript,
                ast.Compare,
                ast.BoolOp,
                ast.IfExp,
                ast.Dict,
                ast.List,
                ast.Set,
                ast.Lambda,
                ast.ListComp,
                ast.DictComp,
                ast.GeneratorExp,
                ast.Await,
                ast.Yield,
                ast.YieldFrom,
                ast.Import,
                ast.ImportFrom,
                ast.Global,
                ast.Nonlocal,
                ast.With,
                ast.Try,
                ast.While,
                ast.For,
                ast.Assign,
                ast.AnnAssign,
                ast.AugAssign,
                ast.FunctionDef,
                ast.ClassDef,
                ast.Return,
            ),
        ):
            raise UnsafeFormulaError(f"Запрещённая конструкция: {type(n).__name__}")

        # Прочие узлы (Load/Store и пр.) игнорируем.


def _make_eval_env(
    *,
    n: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Окружение функций для формул."""

    def randn(scale: float = 1.0) -> np.ndarray:
        return rng.normal(0.0, float(scale), size=n)

    def randu(scale: float = 1.0) -> np.ndarray:
        return rng.uniform(-float(scale), float(scale), size=n)

    def rw(scale: float = 1.0) -> np.ndarray:
        return np.cumsum(randn(scale))

    def ar1(phi: float = 0.7, scale: float = 1.0) -> np.ndarray:
        e = randn(scale)
        x = np.zeros(n, dtype=float)
        p = float(phi)
        for i in range(1, n):
            x[i] = p * x[i - 1] + e[i]
        return x

    # numpy ufuncs
    env: Dict[str, Any] = {
        "pi": float(np.pi),
        "e": float(np.e),
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "clip": np.clip,
        "where": np.where,
        "minimum": np.minimum,
        "maximum": np.maximum,
        "randn": randn,
        "randu": randu,
        "rw": rw,
        "ar1": ar1,
    }
    return env


def safe_eval_vector(expr: str, *, env: Mapping[str, Any], names: Mapping[str, Any]) -> np.ndarray:
    """Вычисляет формулу как вектор длины N в безопасном окружении."""
    expr = (expr or "").strip()
    if not expr:
        raise ValueError("Пустая формула")

    # Разрешаем имена (t, x, y, z, ... + константы) и функции из env.
    allowed_funcs = {k for k, v in env.items() if callable(v)}
    allowed_names = set(names.keys()) | {k for k, v in env.items() if not callable(v)}

    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Синтаксическая ошибка в формуле: {e}") from e

    _validate_ast(node, allowed_names=allowed_names, allowed_funcs=allowed_funcs)

    code = compile(node, "<formula>", "eval")
    out = eval(code, {"__builtins__": {}}, {**env, **names})  # noqa: S307

    arr = np.asarray(out, dtype=float)
    if arr.shape == ():
        # скаляр -> растягиваем
        arr = np.full((int(len(names["t"])),), float(arr), dtype=float)

    if arr.shape[0] != len(names["t"]):
        raise ValueError(f"Формула вернула массив длины {arr.shape[0]}, ожидалась {len(names['t'])}")
    return arr


def generate_formula_dataset(
    *,
    n_samples: int = 500,
    dt: float = 1.0,
    seed: int | None = 42,
    specs: Iterable[FormulaSpec] | None = None,
) -> pd.DataFrame:
    """Генерирует датасет по формулам.

    Семантика зависимостей:
    - первый ряд может использовать только t
    - следующий может использовать t и ранее вычисленные ряды (x, y, ...)

    Пример:
      X: sin(2*pi*t/50) + 0.2*randn()
      Y: 0.8*X + 0.3*randn()
      Z: rw(0.5)

    Переменные в формулах:
      t — массив времени длины N
      X, Y, Z ... — ранее вычисленные ряды (регистр важен: используйте имена рядов)

    Разрешённые функции:
      sin, cos, tan, exp, log, sqrt, abs, clip, where, minimum, maximum,
      randn(scale=1), randu(scale=1), rw(scale=1), ar1(phi=0.7, scale=1)
    """

    if specs is None:
        specs = [
            FormulaSpec("X", "sin(2*pi*t/50) + 0.2*randn()"),
            FormulaSpec("Y", "0.8*X + 0.3*randn()"),
            FormulaSpec("Z", "rw(0.5)"),
        ]

    n = int(n_samples)
    if n < 5:
        raise ValueError("n_samples должно быть >= 5")

    t = np.arange(n, dtype=float) * float(dt)
    rng = np.random.default_rng(seed)
    env = _make_eval_env(n=n, rng=rng)

    names: Dict[str, Any] = {"t": t}
    out: Dict[str, np.ndarray] = {}

    for spec in specs:
        if not spec.name or not spec.expr:
            continue
        vec = safe_eval_vector(spec.expr, env=env, names={**names, **out})
        out[str(spec.name)] = vec

    if not out:
        raise ValueError("Не удалось сгенерировать ни одного ряда")

    return pd.DataFrame(out)
