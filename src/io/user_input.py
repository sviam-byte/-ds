from __future__ import annotations

"""Парсинг пользовательского ввода и нормализация параметров запуска.

Модуль специально изолирован от вычислительного ядра: он только превращает
свободный пользовательский ввод в строгую структуру `RunSpec`.

Поддержка "сканов" (window_pos/window_size/lag/cube) вынесена сюда же:
CLI/Web/GUI могут по-разному собирать ввод, но нормализация должна быть общей.
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


def _parse_bool(v: Any, default: bool) -> bool:
    """Нормализует bool из разных пользовательских представлений."""
    if v is None:
        return default
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, (int, float)):
        return bool(int(v) != 0)
    s = str(v).strip().lower()
    if not s:
        return default
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _parse_int(v: Any, default: int, *, min_v: int | None = None, max_v: int | None = None) -> int:
    """Нормализует int с дефолтом и опциональным clamping границ."""
    try:
        x = int(v)
    except Exception:
        x = int(default)
    if min_v is not None:
        x = max(int(min_v), x)
    if max_v is not None:
        x = min(int(max_v), x)
    return int(x)


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

    # what to compute
    preset: str
    variants: List[str]
    max_lag: int
    lag_selection: str
    lag: int

    # main windowing (affects the returned matrix)
    window_sizes: Optional[List[int]]
    window_stride: Optional[int]
    window_policy: str
    window_cube_level: str
    window_cube_eval_limit: int

    # partial
    partial_mode: str
    pairwise_policy: str
    custom_controls: Optional[List[str]]

    # per-variant overrides (advanced)
    method_options: Optional[Dict[str, Any]]

    # preprocessing
    preprocess: bool
    preprocess_options: Dict[str, Any]

    # scans (independent of main windowing)
    include_scans: bool
    scan_window_pos: bool
    scan_window_size: bool
    scan_lag: bool
    scan_cube: bool

    window_sizes_grid: Optional[List[int]]
    window_min: int
    window_max: int
    window_step: int
    window_size: int

    window_start_min: int
    window_start_max: int
    window_max_windows: int

    lag_grid: Optional[List[int]]
    lag_min: int
    lag_max: int
    lag_step: int

    cube_combo_limit: int
    cube_eval_limit: int
    cube_matrix_mode: str
    cube_matrix_limit: int
    cube_gallery_mode: str
    cube_gallery_k: int
    cube_gallery_limit: int

    # report/output
    output_mode: str
    include_diagnostics: bool
    include_matrix_tables: bool
    include_fft_plots: bool
    harmonic_top_k: int

    def explain(self) -> str:
        """Возвращает человеко-понятное описание параметров запуска."""
        lines: List[str] = []
        lines.append(f"preset={self.preset}")
        lines.append(f"variants={','.join(self.variants) if self.variants else '—'}")
        lines.append(f"lag_selection={self.lag_selection} (max_lag={self.max_lag}, lag={self.lag})")

        if self.window_sizes:
            stride = self.window_stride if self.window_stride is not None else "auto"
            lines.append(f"main_windows={self.window_sizes} stride={stride} policy={self.window_policy}")
        else:
            lines.append("main_windows=off")

        lines.append(f"partial_mode={self.partial_mode}, pairwise_policy={self.pairwise_policy}")
        if self.custom_controls:
            lines.append(f"custom_controls={self.custom_controls}")

        lines.append(f"preprocess={'on' if self.preprocess else 'off'} options={self.preprocess_options or {}}")
        lines.append(f"main_window_cube={self.window_cube_level} (eval_limit={self.window_cube_eval_limit})")

        lines.append(f"output_mode={self.output_mode}")
        lines.append(
            "html: "
            + f"diagnostics={'on' if self.include_diagnostics else 'off'}, "
            + f"scans={'on' if self.include_scans else 'off'}, "
            + f"matrix_tables={'on' if self.include_matrix_tables else 'off'}, "
            + f"fft_plots={'on' if self.include_fft_plots else 'off'} (harmonic_top_k={self.harmonic_top_k})"
        )

        if self.include_scans:
            flags = {
                "window_pos": self.scan_window_pos,
                "window_size": self.scan_window_size,
                "lag": self.scan_lag,
                "cube": self.scan_cube,
            }
            lines.append(f"scans_flags={flags}")
            if self.window_sizes_grid:
                lines.append(f"window_sizes_grid={self.window_sizes_grid}")
            else:
                lines.append(f"window_grid=min:{self.window_min} max:{self.window_max} step:{self.window_step}")
            lines.append(
                f"window_pos: w={self.window_size} start=[{self.window_start_min},{self.window_start_max}] max_windows={self.window_max_windows}"
            )
            if self.lag_grid:
                lines.append(f"lag_grid={self.lag_grid}")
            else:
                lines.append(f"lag_grid=min:{self.lag_min} max:{self.lag_max} step:{self.lag_step}")
            lines.append(
                f"cube: combos_limit={self.cube_combo_limit} eval_limit={self.cube_eval_limit} matrix_mode={self.cube_matrix_mode} matrix_limit={self.cube_matrix_limit}"
            )

        if self.method_options:
            lines.append(f"method_options={list(self.method_options.keys())}")

        return "\n".join(lines)


def build_run_spec(user_cfg: Dict[str, Any], *, default_max_lag: int = 12) -> RunSpec:
    """Собирает `RunSpec` из словаря пользовательского ввода."""
    preset = str(user_cfg.get("preset", "basic")).strip().lower()

    raw_variants = user_cfg.get("variants")
    variants_list = _parse_list(raw_variants)
    if not variants_list:
        variants_list = [preset]

    max_lag = _parse_int(user_cfg.get("max_lag", default_max_lag), default_max_lag, min_v=1)
    lag_selection = str(user_cfg.get("lag_selection", "optimize")).strip().lower()
    if lag_selection not in {"optimize", "fixed"}:
        lag_selection = "optimize"
    lag = _parse_int(user_cfg.get("lag", 1), 1, min_v=1)

    # --- main windowing ---
    window_sizes = _parse_int_list(user_cfg.get("window_sizes"))
    window_stride_raw = user_cfg.get("window_stride")
    window_stride = int(window_stride_raw) if window_stride_raw is not None and str(window_stride_raw).strip() else None

    window_policy = str(user_cfg.get("window_policy", "best")).strip().lower()
    if window_policy not in {"best", "mean", "none", "off"}:
        window_policy = "best"

    # --- partial ---
    partial_mode = str(user_cfg.get("partial_mode", "pairwise")).strip().lower()
    pairwise_policy = str(user_cfg.get("pairwise_policy", "others")).strip().lower()
    custom_controls = _parse_list(user_cfg.get("custom_controls")) or None

    # --- preprocessing ---
    preprocess = _parse_bool(user_cfg.get("preprocess", True), True)
    preprocess_options = user_cfg.get("preprocess_options") or {}
    if not isinstance(preprocess_options, dict):
        preprocess_options = {}

    # --- legacy: main-window cube search (affects chosen lag / best window) ---
    window_cube_level = str(user_cfg.get("window_cube", user_cfg.get("window_cube_level", "off"))).strip().lower()
    if window_cube_level not in {"off", "basic", "full"}:
        window_cube_level = "off"
    window_cube_eval_limit = _parse_int(
        user_cfg.get("window_cube_eval_limit", 120 if window_cube_level == "full" else 60),
        120 if window_cube_level == "full" else 60,
        min_v=20,
        max_v=2000,
    )

    # --- report/output ---
    output_mode = str(user_cfg.get("output_mode", "both")).strip().lower()
    if output_mode not in {"html", "excel", "both"}:
        output_mode = "both"

    include_diagnostics = _parse_bool(user_cfg.get("include_diagnostics", True), True)
    include_scans = _parse_bool(user_cfg.get("include_scans", True), True)
    include_matrix_tables = _parse_bool(user_cfg.get("include_matrix_tables", False), False)

    include_fft_plots = _parse_bool(user_cfg.get("include_fft_plots", False), False)
    harmonic_top_k = _parse_int(user_cfg.get("harmonic_top_k", 5), 5, min_v=1, max_v=20)

    # --- scans flags ---
    scan_window_pos = _parse_bool(user_cfg.get("scan_window_pos", True), True)
    scan_window_size = _parse_bool(user_cfg.get("scan_window_size", True), True)
    scan_lag = _parse_bool(user_cfg.get("scan_lag", True), True)
    scan_cube = _parse_bool(user_cfg.get("scan_cube", True), True)
    if not include_scans:
        scan_window_pos = scan_window_size = scan_lag = scan_cube = False

    # window grids
    window_sizes_grid = _parse_int_list(user_cfg.get("window_sizes_grid"))
    window_min = _parse_int(user_cfg.get("window_min", 64), 64, min_v=2)
    window_max = _parse_int(user_cfg.get("window_max", 192), 192, min_v=2)
    if window_max < window_min:
        window_max = window_min
    window_step = _parse_int(user_cfg.get("window_step", 64), 64, min_v=1)

    window_size = _parse_int(user_cfg.get("window_size", (window_min + window_max) // 2), (window_min + window_max) // 2, min_v=2)

    window_start_min = _parse_int(user_cfg.get("window_start_min", 0), 0, min_v=0)
    window_start_max = _parse_int(user_cfg.get("window_start_max", 0), 0, min_v=0)
    window_max_windows = _parse_int(user_cfg.get("window_max_windows", 60), 60, min_v=1, max_v=5000)

    # lag grid
    lag_grid = _parse_int_list(user_cfg.get("lag_grid"))
    lag_min = _parse_int(user_cfg.get("lag_min", 1), 1, min_v=1)
    lag_max_default = min(max_lag, 3) if max_lag >= 1 else 3
    lag_max = _parse_int(user_cfg.get("lag_max", lag_max_default), lag_max_default, min_v=lag_min)
    lag_step = _parse_int(user_cfg.get("lag_step", 1), 1, min_v=1)

    # cube controls
    cube_combo_limit = _parse_int(user_cfg.get("cube_combo_limit", 9), 9, min_v=1, max_v=100000)
    cube_eval_limit = _parse_int(user_cfg.get("cube_eval_limit", 225), 225, min_v=1, max_v=10000000)
    cube_matrix_mode = str(user_cfg.get("cube_matrix_mode", "all")).strip().lower()
    if cube_matrix_mode not in {"selected", "all"}:
        cube_matrix_mode = "all"
    cube_matrix_limit = _parse_int(user_cfg.get("cube_matrix_limit", cube_eval_limit), cube_eval_limit, min_v=1, max_v=10000000)

    cube_gallery_mode = str(user_cfg.get("cube_gallery_mode", "extremes") or "extremes").strip().lower()
    if cube_gallery_mode not in {"extremes", "topbottom", "quantiles"}:
        cube_gallery_mode = "extremes"
    cube_gallery_k = _parse_int(user_cfg.get("cube_gallery_k", 1), 1, min_v=1, max_v=1000)
    cube_gallery_limit = _parse_int(user_cfg.get("cube_gallery_limit", 60), 60, min_v=3, max_v=5000)

    # per-method overrides (advanced)
    method_options = user_cfg.get("method_options")
    if not isinstance(method_options, dict):
        method_options = None

    return RunSpec(
        preset=preset,
        variants=variants_list,
        max_lag=max_lag,
        lag_selection=lag_selection,
        lag=lag,
        window_sizes=window_sizes,
        window_stride=window_stride,
        window_policy=window_policy,
        window_cube_level=window_cube_level,
        window_cube_eval_limit=window_cube_eval_limit,
        partial_mode=partial_mode,
        pairwise_policy=pairwise_policy,
        custom_controls=custom_controls,
        method_options=method_options,
        preprocess=preprocess,
        preprocess_options=preprocess_options,
        include_scans=include_scans,
        scan_window_pos=scan_window_pos,
        scan_window_size=scan_window_size,
        scan_lag=scan_lag,
        scan_cube=scan_cube,
        window_sizes_grid=window_sizes_grid,
        window_min=window_min,
        window_max=window_max,
        window_step=window_step,
        window_size=window_size,
        window_start_min=window_start_min,
        window_start_max=window_start_max,
        window_max_windows=window_max_windows,
        lag_grid=lag_grid,
        lag_min=lag_min,
        lag_max=lag_max,
        lag_step=lag_step,
        cube_combo_limit=cube_combo_limit,
        cube_eval_limit=cube_eval_limit,
        cube_matrix_mode=cube_matrix_mode,
        cube_matrix_limit=cube_matrix_limit,
        cube_gallery_mode=cube_gallery_mode,
        cube_gallery_k=cube_gallery_k,
        cube_gallery_limit=cube_gallery_limit,
        output_mode=output_mode,
        include_diagnostics=include_diagnostics,
        include_matrix_tables=include_matrix_tables,
        include_fft_plots=include_fft_plots,
        harmonic_top_k=harmonic_top_k,
    )
