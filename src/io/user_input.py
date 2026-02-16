from __future__ import annotations

"""Парсинг пользовательского ввода и нормализация параметров запуска.

Модуль специально изолирован от вычислительного ядра: он только превращает
свободный пользовательский ввод в строгую структуру `RunSpec`.
"""

import ast
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _split_kv(text: str) -> Dict[str, str]:
    """Разбирает простой формат `key=value; key2=value2`.

    Также поддерживает переносы строк вместо `;` и одиночное слово
    (трактуется как `preset=<word>`).
    """
    out: Dict[str, str] = {}
    if not text:
        return out

    parts: List[str] = []
    for chunk in text.replace("\n", ";").split(";"):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)

    for item in parts:
        if "=" not in item:
            out.setdefault("preset", item.strip())
            continue
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _parse_list(v: Any) -> List[str]:
    """Нормализует значение в список строк."""
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x).strip() for x in v if str(x).strip()]

    s = str(v).strip()
    if not s:
        return []

    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    return [x.strip() for x in s.split() if x.strip()]


def _parse_int_list(v: Any) -> Optional[List[int]]:
    """Преобразует значение в список int, пропуская невалидные токены."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        xs = []
        for x in v:
            try:
                xs.append(int(x))
            except Exception:
                continue
        return xs or None

    s = str(v).strip()
    if not s:
        return None

    xs = []
    for tok in s.replace("[", "").replace("]", "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            xs.append(int(tok))
        except Exception:
            continue
    return xs or None


def parse_user_input(text: str) -> Dict[str, Any]:
    """Парсит пользовательскую строку в словарь.

    Поддерживаемые форматы:
    - пустая строка -> {}
    - JSON-словарь
    - Python dict literal (`ast.literal_eval`)
    - key=value; key2=value2
    - одиночное слово -> `{"preset": "..."}`
    """
    text = (text or "").strip()
    if not text:
        return {}

    if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    if text.startswith("{") and text.endswith("}"):
        try:
            obj = ast.literal_eval(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return _split_kv(text)


@dataclass
class RunSpec:
    """Нормализованная спецификация запуска анализа."""

    preset: str
    variants: List[str]
    max_lag: int
    lag_selection: str
    window_sizes: Optional[List[int]]
    window_stride: Optional[int]
    window_policy: str
    partial_mode: str
    pairwise_policy: str
    custom_controls: Optional[List[str]]

    # UI/report options
    preprocess: bool
    preprocess_options: Dict[str, Any]
    window_cube_level: str
    harmonic_top_k: int

    def explain(self) -> str:
        """Возвращает человеко-понятное описание параметров запуска."""
        lines: List[str] = []
        lines.append(f"preset={self.preset}")
        lines.append(f"variants={','.join(self.variants) if self.variants else '—'}")
        lines.append(f"lag_selection={self.lag_selection} (max_lag={self.max_lag})")
        if self.window_sizes:
            stride = self.window_stride if self.window_stride is not None else "auto"
            lines.append(f"windows={self.window_sizes} stride={stride} policy={self.window_policy}")
        else:
            lines.append("windows=off")
        lines.append(f"partial_mode={self.partial_mode}, pairwise_policy={self.pairwise_policy}")
        if self.custom_controls:
            lines.append(f"custom_controls={self.custom_controls}")
        lines.append(f"preprocess={'on' if self.preprocess else 'off'} options={self.preprocess_options or {}}")
        lines.append(f"window_cube_level={self.window_cube_level}")
        lines.append(f"harmonic_top_k={self.harmonic_top_k}")
        return "\n".join(lines)


def build_run_spec(user_cfg: Dict[str, Any], *, default_max_lag: int = 12) -> RunSpec:
    """Собирает `RunSpec` из словаря пользовательского ввода."""
    preset = str(user_cfg.get("preset", "basic")).strip().lower()

    raw_variants = user_cfg.get("variants")
    variants_list = _parse_list(raw_variants)
    if not variants_list:
        variants_list = [preset]

    max_lag = max(1, int(user_cfg.get("max_lag", default_max_lag)))
    lag_selection = str(user_cfg.get("lag_selection", "optimize")).strip().lower()

    window_sizes = _parse_int_list(user_cfg.get("window_sizes"))
    window_stride_raw = user_cfg.get("window_stride")
    window_stride = int(window_stride_raw) if window_stride_raw is not None and str(window_stride_raw).strip() else None
    window_policy = str(user_cfg.get("window_policy", "best")).strip().lower()

    partial_mode = str(user_cfg.get("partial_mode", "pairwise")).strip().lower()
    pairwise_policy = str(user_cfg.get("pairwise_policy", "others")).strip().lower()
    custom_controls = _parse_list(user_cfg.get("custom_controls")) or None

    preprocess = bool(user_cfg.get("preprocess", True))
    preprocess_options = user_cfg.get("preprocess_options") or {}
    if not isinstance(preprocess_options, dict):
        preprocess_options = {}

    window_cube_level = str(user_cfg.get("window_cube_level", "off")).strip().lower()
    if window_cube_level not in {"off", "basic", "full"}:
        window_cube_level = "off"

    harmonic_top_k = int(user_cfg.get("harmonic_top_k", 5))
    harmonic_top_k = max(1, min(20, harmonic_top_k))

    return RunSpec(
        preset=preset,
        variants=variants_list,
        max_lag=max_lag,
        lag_selection=lag_selection,
        window_sizes=window_sizes,
        window_stride=window_stride,
        window_policy=window_policy,
        partial_mode=partial_mode,
        pairwise_policy=pairwise_policy,
        custom_controls=custom_controls,
        preprocess=preprocess,
        preprocess_options=preprocess_options,
        window_cube_level=window_cube_level,
        harmonic_top_k=harmonic_top_k,
    )
