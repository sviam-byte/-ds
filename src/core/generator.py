"""
Модуль генерации синтетических временных рядов для тестирования.

Цель: быстро собирать синтетические датасеты, где:
- каждая переменная может быть шумом/моделью/функцией времени;
- каждая переменная может зависеть от других (через лаг >= 1);
- всё детерминируемо по seed.

Дополнительно сохранён старый API формульного генератора для обратной совместимости.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional

import numpy as np
import pandas as pd


# ----------------------------
# Пресеты (обратная совместимость)
# ----------------------------
def generate_coupled_system(
    n_samples: int = 500,
    coupling_strength: float = 0.8,
    noise_level: float = 0.2,
) -> pd.DataFrame:
    """Генерирует систему из 4 переменных:

    - X: AR(1) (источник).
    - Y: зависит от X (X -> Y) с лагом 1.
    - Z: независимый random walk.
    - S: сезонный синус.
    """
    rng = np.random.default_rng(42)

    e_x = rng.normal(0, 1, n_samples)
    e_y = rng.normal(0, 1, n_samples)
    e_z = rng.normal(0, 1, n_samples)

    x = np.zeros(n_samples, dtype=float)
    y = np.zeros(n_samples, dtype=float)

    for t in range(1, n_samples):
        x[t] = 0.5 * x[t - 1] + noise_level * e_x[t]
        y[t] = 0.5 * y[t - 1] + coupling_strength * x[t - 1] + noise_level * e_y[t]

    z = np.cumsum(e_z * noise_level)

    t_idx = np.arange(n_samples, dtype=float)
    s = np.sin(2 * np.pi * t_idx / 50.0) + rng.normal(0, 0.1, n_samples)

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
    rng = np.random.default_rng()
    data: Dict[str, np.ndarray] = {}
    for i in range(int(max(1, n_vars))):
        data[f"RW_{i + 1}"] = np.cumsum(rng.normal(0, 1, n_samples))
    return pd.DataFrame(data)


# ----------------------------
# Конструктор синтетики (новый API)
# ----------------------------
BaseType = Literal["white", "ar1", "rw", "sin", "cos", "linear", "const"]


@dataclass(slots=True)
class CouplingTerm:
    """Линейная зависимость от другой переменной с лагом >= 1: coef * src[t-lag]."""

    src: str
    coef: float
    lag: int = 1


@dataclass(slots=True)
class SeriesSpec:
    """Спецификация одной переменной для генерации."""

    name: str
    base: BaseType = "white"
    params: Dict[str, float] = field(default_factory=dict)
    noise: float = 0.0
    couplings: List[CouplingTerm] = field(default_factory=list)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def generate_synthetic_dataset(
    specs: List[SeriesSpec],
    *,
    n_samples: int = 800,
    dt: float = 1.0,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """Генерация мультивариантного датасета по списку ``SeriesSpec``."""
    n = int(max(10, n_samples))
    dt = float(dt) if dt is not None else 1.0
    if not np.isfinite(dt) or dt <= 0:
        dt = 1.0

    rng = np.random.default_rng(seed)

    names = [s.name for s in specs]
    out: Dict[str, np.ndarray] = {nm: np.zeros(n, dtype=float) for nm in names}
    t = np.arange(n, dtype=float) * dt

    for i in range(n):
        for s in specs:
            prev = out[s.name][i - 1] if i > 0 else 0.0
            base = 0.0
            p = s.params or {}

            if s.base == "white":
                base = rng.normal(0.0, _safe_float(p.get("scale", 1.0), 1.0))
            elif s.base == "ar1":
                phi = _safe_float(p.get("phi", 0.5), 0.5)
                sc = _safe_float(p.get("scale", 1.0), 1.0)
                base = phi * prev + rng.normal(0.0, sc)
            elif s.base == "rw":
                sc = _safe_float(p.get("scale", 1.0), 1.0)
                base = prev + rng.normal(0.0, sc)
            elif s.base in {"sin", "cos"}:
                amp = _safe_float(p.get("amp", 1.0), 1.0)
                per = _safe_float(p.get("period", 50.0), 50.0)
                if per <= 0:
                    per = 50.0
                ph = _safe_float(p.get("phase", 0.0), 0.0)
                bn = _safe_float(p.get("base_noise", 0.0), 0.0)
                arg = 2.0 * np.pi * (t[i] / per) + ph
                wave = np.sin(arg) if s.base == "sin" else np.cos(arg)
                base = amp * wave + (rng.normal(0.0, bn) if bn > 0 else 0.0)
            elif s.base == "linear":
                slope = _safe_float(p.get("slope", 0.0), 0.0)
                intercept = _safe_float(p.get("intercept", 0.0), 0.0)
                bn = _safe_float(p.get("base_noise", 0.0), 0.0)
                base = intercept + slope * t[i] + (rng.normal(0.0, bn) if bn > 0 else 0.0)
            elif s.base == "const":
                val = _safe_float(p.get("value", 0.0), 0.0)
                bn = _safe_float(p.get("base_noise", 0.0), 0.0)
                base = val + (rng.normal(0.0, bn) if bn > 0 else 0.0)

            coupling_sum = 0.0
            for c in (s.couplings or []):
                lag = int(max(1, c.lag))
                if i - lag < 0:
                    continue
                src = str(c.src)
                if src not in out:
                    continue
                coupling_sum += float(c.coef) * float(out[src][i - lag])

            extra = rng.normal(0.0, float(s.noise)) if (s.noise is not None and float(s.noise) > 0) else 0.0
            out[s.name][i] = float(base + coupling_sum + extra)

    return pd.DataFrame(out)


def build_specs_from_dict(builder: Dict) -> List[SeriesSpec]:
    """Преобразует словарь builder в список ``SeriesSpec``."""
    specs: List[SeriesSpec] = []
    series_list = builder.get("series") if isinstance(builder, dict) else None
    if not isinstance(series_list, list):
        raise ValueError("builder['series'] must be a list")

    for item in series_list:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        base = str(item.get("base", "white"))
        if base not in {"white", "ar1", "rw", "sin", "cos", "linear", "const"}:
            base = "white"
        params = item.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        noise = _safe_float(item.get("noise", 0.0), 0.0)

        couplings: List[CouplingTerm] = []
        for c in (item.get("couplings") or []):
            if not isinstance(c, dict):
                continue
            src = str(c.get("src", "")).strip()
            if not src:
                continue
            coef = _safe_float(c.get("coef", 0.0), 0.0)
            lag = int(max(1, int(_safe_float(c.get("lag", 1), 1))))
            if coef == 0.0:
                continue
            couplings.append(CouplingTerm(src=src, coef=coef, lag=lag))

        specs.append(
            SeriesSpec(
                name=name,
                base=base,
                params={k: _safe_float(v, 0.0) for k, v in params.items()},
                noise=noise,
                couplings=couplings,
            )
        )

    if len(specs) < 1:
        raise ValueError("no valid series specs in builder")
    return specs


def generate_from_builder(
    builder: Dict,
    *,
    n_samples: int = 800,
    dt: float = 1.0,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """High-level API: ``builder(dict) -> DataFrame``."""
    specs = build_specs_from_dict(builder)
    return generate_synthetic_dataset(specs, n_samples=n_samples, dt=dt, seed=seed)


# =========================
# Генерация по формулам (legacy API)
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
            continue
        if isinstance(n, ast.keyword):
            if n.arg is None:
                raise UnsafeFormulaError("Распаковка **kwargs в вызовах запрещена")
            continue

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


def _make_eval_env(*, n: int, rng: np.random.Generator) -> Dict[str, Any]:
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

    return {
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


def safe_eval_vector(expr: str, *, env: Mapping[str, Any], names: Mapping[str, Any]) -> np.ndarray:
    """Вычисляет формулу как вектор длины N в безопасном окружении."""
    expr = (expr or "").strip()
    if not expr:
        raise ValueError("Пустая формула")

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
        arr = np.full((int(len(names["t"])),), float(arr), dtype=float)

    if arr.ndim != 1:
        raise ValueError(f"Формула должна возвращать 1D-массив, получено ndim={arr.ndim}")

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
    """Генератор датасета по формулам (legacy API)."""
    n = int(max(1, n_samples))
    t = np.arange(n, dtype=float) * float(dt)
    rng = np.random.default_rng(seed)

    specs = list(specs or [FormulaSpec("X", "randn()"), FormulaSpec("Y", "randn()"), FormulaSpec("Z", "randn()")])
    if len(specs) == 0:
        raise ValueError("Не задано ни одной формулы")

    env = _make_eval_env(n=n, rng=rng)
    cols: Dict[str, np.ndarray] = {}

    for idx, spec in enumerate(specs):
        name = (spec.name or "").strip() or f"V{idx+1}"
        names: Dict[str, Any] = {"t": t}
        for prev_name, prev_values in cols.items():
            names[prev_name] = prev_values
            names[prev_name.lower()] = prev_values
        arr = safe_eval_vector(spec.expr, env=env, names=names)
        cols[name] = arr

    return pd.DataFrame(cols)
