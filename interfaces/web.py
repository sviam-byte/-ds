"""Веб-интерфейс (Streamlit) для Time Series Analysis Tool (локально).

Фокус: понятный русский ввод + сразу видно, что будет посчитано и что получится на выходе.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import EXPERIMENTAL_METHODS, SAVE_FOLDER, STABLE_METHODS
from src.core import engine
from src.core import generator as synth


def _parse_int_list_text(text: str) -> list[int] | None:
    """Парсит строку формата `1,2,3` в список целых значений."""
    text = (text or "").strip()
    if not text:
        return None
    xs: list[int] = []
    for tok in text.replace("[", "").replace("]", "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            xs.append(int(tok))
        except Exception:
            continue
    return xs or None


def _make_run_dir(prefix: str = "run") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(SAVE_FOLDER) / "runs" / f"{prefix}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _sidebar_plan(
    *,
    source_label: str,
    dataset_label: str,
    n_points: Optional[int],
    n_vars: Optional[int],
    methods: list[str],
    output_mode: str,
    include_scans: bool,
    include_diagnostics: bool,
    include_matrix_tables: bool,
    scan_flags: Dict[str, bool],
) -> None:
    st.sidebar.markdown("### План вывода")
    st.sidebar.caption("Что будет сделано и какие файлы появятся.")

    st.sidebar.markdown(f"**Источник:** {source_label}")
    st.sidebar.markdown(f"**Датасет:** {dataset_label}")
    if n_points is not None and n_vars is not None:
        st.sidebar.markdown(f"**Размер:** {n_points} точек × {n_vars} рядов")

    st.sidebar.markdown("**Методы:**")
    if methods:
        st.sidebar.write(", ".join(methods))
    else:
        st.sidebar.write("—")

    st.sidebar.markdown("**Сканы (в отчёте):**")
    if include_scans:
        st.sidebar.write(", ".join([k for k, v in scan_flags.items() if v]) or "—")
    else:
        st.sidebar.write("выключены")

    st.sidebar.markdown("**HTML-опции:**")
    st.sidebar.write(
        "диагностика=" + ("да" if include_diagnostics else "нет")
        + " • сканы=" + ("да" if include_scans else "нет")
        + " • таблица матрицы=" + ("да" if include_matrix_tables else "нет")
    )

    st.sidebar.markdown("**Выходные файлы:**")
    files = []
    if output_mode in {"html", "both"}:
        files.append("report.html")
    if output_mode in {"excel", "both"}:
        files.append("report.xlsx")
    files.append("series.xlsx")
    st.sidebar.write(", ".join(files))


def _ui_series_spec(name: str, other_names: list[str]) -> dict:
    """UI-формочка одной переменной для конструктора синтетики."""
    st.markdown(f"**{name}**")
    base_map = {
        "Белый шум": "white",
        "AR(1)": "ar1",
        "Случайное блуждание (RW)": "rw",
        "Синус": "sin",
        "Косинус": "cos",
        "Линейный тренд": "linear",
        "Константа": "const",
    }
    base_ui = st.selectbox(
        f"{name}: базовый тип",
        list(base_map.keys()),
        index=0,
        key=f"{name}_base",
    )
    base = base_map[base_ui]

    params: Dict[str, float] = {}
    if base == "white":
        params["scale"] = float(st.number_input(f"{name}: sigma", value=1.0, step=0.1, key=f"{name}_scale"))
    elif base == "ar1":
        params["phi"] = float(
            st.number_input(f"{name}: phi", min_value=-0.99, max_value=0.99, value=0.6, step=0.01, key=f"{name}_phi")
        )
        params["scale"] = float(st.number_input(f"{name}: sigma", value=0.5, step=0.1, key=f"{name}_scale"))
    elif base == "rw":
        params["scale"] = float(st.number_input(f"{name}: step sigma", value=0.5, step=0.1, key=f"{name}_scale"))
    elif base in {"sin", "cos"}:
        params["amp"] = float(st.number_input(f"{name}: amplitude", value=1.0, step=0.1, key=f"{name}_amp"))
        params["period"] = float(
            st.number_input(f"{name}: period", min_value=2.0, value=50.0, step=1.0, key=f"{name}_per")
        )
        params["phase"] = float(st.number_input(f"{name}: phase (rad)", value=0.0, step=0.1, key=f"{name}_ph"))
        params["base_noise"] = float(st.number_input(f"{name}: base noise", value=0.0, step=0.05, key=f"{name}_bn"))
    elif base == "linear":
        params["slope"] = float(st.number_input(f"{name}: slope", value=0.0, step=0.01, key=f"{name}_sl"))
        params["intercept"] = float(st.number_input(f"{name}: intercept", value=0.0, step=0.1, key=f"{name}_ic"))
        params["base_noise"] = float(st.number_input(f"{name}: base noise", value=0.0, step=0.05, key=f"{name}_bn"))
    elif base == "const":
        params["value"] = float(st.number_input(f"{name}: value", value=0.0, step=0.1, key=f"{name}_val"))
        params["base_noise"] = float(st.number_input(f"{name}: base noise", value=0.0, step=0.05, key=f"{name}_bn"))

    noise = float(st.number_input(f"{name}: добавочный независимый шум", value=0.0, step=0.05, key=f"{name}_noise"))

    st.caption("Зависимости: coef * src[t-lag]. Лаг ≥ 1, чтобы система была однозначной.")
    couplings = []
    for src in other_names:
        use = st.checkbox(f"{name} зависит от {src}", value=False, key=f"{name}_use_{src}")
        if use:
            coef = float(st.number_input(f"{name} <- {src}: coef", value=0.5, step=0.05, key=f"{name}_coef_{src}"))
            lag = int(
                st.number_input(
                    f"{name} <- {src}: lag",
                    min_value=1,
                    max_value=200,
                    value=1,
                    step=1,
                    key=f"{name}_lag_{src}",
                )
            )
            if coef != 0.0:
                couplings.append({"src": src, "coef": coef, "lag": lag})

    return {"name": name, "base": base, "params": params, "noise": noise, "couplings": couplings}


def main() -> None:
    st.set_page_config(page_title="Анализ временных рядов", layout="wide")
    st.title("Анализ связности временных рядов")

    # --- Источник данных ---
    source = st.radio("Источник данных", ["Файл (CSV/XLSX)", "Синтетика"], index=0, horizontal=True)
    uploaded_file = None
    synth_df: Optional[pd.DataFrame] = None
    synth_name = "synthetic"

    if source.startswith("Файл"):
        uploaded_file = st.file_uploader("Загрузите CSV или Excel", type=["csv", "xlsx"])
    else:
        with st.expander("Синтетика: конструктор X/Y/Z", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                n_samples = int(st.number_input("Длина ряда (n_samples)", min_value=50, max_value=20000, value=800, step=50))
            with c2:
                seed = int(st.number_input("Seed", min_value=0, max_value=10_000_000, value=42, step=1))
            with c3:
                dt = float(st.number_input("dt (шаг времени)", min_value=0.0001, value=1.0, step=0.1, format="%.4f"))

            st.markdown("---")
            colA, colB, colC = st.columns(3)
            with colA:
                spec_x = _ui_series_spec("X", ["Y", "Z"])
            with colB:
                spec_y = _ui_series_spec("Y", ["X", "Z"])
            with colC:
                spec_z = _ui_series_spec("Z", ["X", "Y"])

            builder = {"series": [spec_x, spec_y, spec_z]}
            try:
                synth_df = synth.generate_from_builder(builder, n_samples=n_samples, dt=dt, seed=seed)
                st.caption("Превью синтетики:")
                st.dataframe(synth_df.head(20), use_container_width=True)
                synth_name = f"synth_XYZ_n{n_samples}_seed{seed}"
            except Exception as e:
                st.error(f"Ошибка генерации синтетики: {e}")
                synth_df = None

    # --- Параметры анализа ---
    all_methods = STABLE_METHODS + EXPERIMENTAL_METHODS
    default_methods = STABLE_METHODS[:2] if STABLE_METHODS else all_methods[:2]
    selected_methods = st.multiselect("Методы", all_methods, default=default_methods)

    with st.expander("Параметры запуска", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            lag_selection = st.selectbox("Лаг: выбор", ["optimize", "fixed"], index=0)
            if lag_selection == "fixed":
                lag = st.number_input("lag (если fixed)", min_value=1, max_value=2000, value=1)
            else:
                lag = st.number_input("lag (не используется при optimize)", min_value=1, max_value=2000, value=1)
            max_lag = st.number_input("max_lag (для optimize и сканов)", min_value=1, max_value=2000, value=12)
            alpha = st.number_input("P-value alpha (для Granger/p-value)", 0.0001, 0.5, 0.05, format="%.4f")
            threshold = st.number_input("Порог графа (threshold)", 0.0, 1.0, 0.2, 0.05)

        with col2:
            normalize = st.checkbox("Нормализация (Z-score)", value=True)
            preprocess = st.checkbox("Предобработка (включить)", value=True)
            fill_missing = st.checkbox("Заполнять пропуски (interp)", value=True)
            log_transform = st.checkbox("Лог-преобразование (только >0)", value=False)

            remove_outliers = st.checkbox("Обработка выбросов (Z)", value=True)
            outlier_mode = st.selectbox(
                "Выбросы: режим",
                ["nan (вырезать)", "winsorize (обрезать)"],
                index=0,
                disabled=not remove_outliers,
            )
            outlier_mode_key = "nan" if outlier_mode.startswith("nan") else "winsorize"

            st.markdown("**Доп. предобработка**")
            detrend_linear = st.checkbox("Убрать линейный тренд (detrend)", value=False, disabled=not preprocess)
            deseasonalize = st.checkbox("Убрать сезонность (mean by phase)", value=False, disabled=not preprocess)
            seasonal_period = st.number_input(
                "Период сезонности (если включено)",
                min_value=0,
                max_value=100000,
                value=0,
                step=1,
                disabled=not (preprocess and deseasonalize),
            )
            prewhiten_ar1 = st.checkbox("Убрать AR(1) (prewhiten)", value=False, disabled=not preprocess)

        with col3:
            output_mode = st.selectbox("Выход", ["both", "html", "excel"], index=0)
            include_diagnostics = st.checkbox("HTML: диагностика", value=True)
            include_scans = st.checkbox("HTML: сканы (окна/лаги/куб)", value=True)
            include_matrix_tables = st.checkbox("HTML: таблица матрицы (текстом)", value=False)
            include_fft_plots = st.checkbox("HTML: FFT-графики", value=True)
            harmonic_top_k = st.number_input("Гармоники: top_k", min_value=1, max_value=20, value=5)

        st.markdown("---")
        st.subheader("Сканы (инспектор; не меняют итоговую матрицу)")

        s1, s2, s3 = st.columns(3)
        with s1:
            scan_window_pos = st.checkbox("scan_window_pos", value=True, disabled=not include_scans)
            scan_window_size = st.checkbox("scan_window_size", value=True, disabled=not include_scans)
            scan_lag = st.checkbox("scan_lag", value=True, disabled=not include_scans)
            scan_cube = st.checkbox("scan_cube", value=True, disabled=not include_scans)

        with s2:
            window_min = st.number_input("window_min", min_value=2, max_value=1_000_000, value=64, step=1, disabled=not include_scans)
            window_max = st.number_input("window_max", min_value=2, max_value=1_000_000, value=192, step=1, disabled=not include_scans)
            window_step = st.number_input("window_step", min_value=1, max_value=1_000_000, value=64, step=1, disabled=not include_scans)
            window_size_default = st.number_input(
                "window_size (для scan_window_pos)",
                min_value=2,
                max_value=1_000_000,
                value=128,
                step=1,
                disabled=not include_scans,
            )

        with s3:
            window_start_min = st.number_input(
                "window_start_min (0=auto)",
                min_value=0,
                max_value=10_000_000,
                value=0,
                step=1,
                disabled=not include_scans,
            )
            window_start_max = st.number_input(
                "window_start_max (0=auto)",
                min_value=0,
                max_value=10_000_000,
                value=0,
                step=1,
                disabled=not include_scans,
            )
            window_stride_scan = st.number_input(
                "window_stride (scan, 0=auto)",
                min_value=0,
                max_value=10_000_000,
                value=0,
                step=1,
                disabled=not include_scans,
            )
            window_max_windows = st.number_input(
                "window_max_windows",
                min_value=1,
                max_value=5000,
                value=60,
                step=1,
                disabled=not include_scans,
            )

        st.markdown("**Лаг-сетка (для scan_lag и cube)**")
        l1, l2, l3 = st.columns(3)
        with l1:
            lag_min = st.number_input("lag_min", min_value=1, max_value=20000, value=1, step=1, disabled=not include_scans)
        with l2:
            lag_max = st.number_input(
                "lag_max",
                min_value=1,
                max_value=20000,
                value=min(3, int(max_lag)),
                step=1,
                disabled=not include_scans,
            )
        with l3:
            lag_step = st.number_input("lag_step", min_value=1, max_value=20000, value=1, step=1, disabled=not include_scans)

        st.markdown("**Куб (window_size × lag × position)**")
        k1, k2, k3 = st.columns(3)
        with k1:
            cube_combo_limit = st.number_input("cube_combo_limit", min_value=10, max_value=100000, value=500, step=10, disabled=not include_scans)
            cube_eval_limit = st.number_input("cube_eval_limit", min_value=10, max_value=50000, value=2000, step=10, disabled=not include_scans)
        with k2:
            cube_matrix_mode = st.selectbox("cube_matrix_mode", ["none", "gallery", "all"], index=1, disabled=not include_scans)
            cube_matrix_limit = st.number_input("cube_matrix_limit", min_value=0, max_value=10000, value=300, step=10, disabled=not include_scans)
        with k3:
            cube_gallery_mode = st.selectbox("cube_gallery_mode", ["topk", "quantiles"], index=0, disabled=not include_scans)
            cube_gallery_k = st.number_input("cube_gallery_k", min_value=1, max_value=1000, value=40, step=1, disabled=not include_scans)
            cube_gallery_limit = st.number_input("cube_gallery_limit", min_value=1, max_value=10000, value=300, step=10, disabled=not include_scans)

        st.markdown("**method_options (JSON, опционально)**")
        method_options_text = st.text_area(
            "Пример: {'pearson': {'window': 128}, 'granger': {'maxlag': 8}}",
            value="",
            height=80,
        )

    scan_flags = {
        "window_pos": bool(scan_window_pos and include_scans),
        "window_size": bool(scan_window_size and include_scans),
        "lag": bool(scan_lag and include_scans),
        "cube": bool(scan_cube and include_scans),
    }

    _sidebar_plan(
        source_label=("Файл" if source.startswith("Файл") else "Синтетика"),
        dataset_label=(uploaded_file.name if uploaded_file else synth_name),
        n_points=(len(synth_df) if synth_df is not None else None),
        n_vars=(len(synth_df.columns) if synth_df is not None else None),
        methods=selected_methods,
        output_mode=output_mode,
        include_scans=include_scans,
        include_diagnostics=include_diagnostics,
        include_matrix_tables=include_matrix_tables,
        scan_flags=scan_flags,
    )

    if st.button("Запустить анализ", type="primary"):
        if source.startswith("Файл") and not uploaded_file:
            st.error("Файл не загружен.")
            return
        if source.startswith("Синтетика") and synth_df is None:
            st.error("Синтетика не сгенерирована.")
            return
        if not selected_methods:
            st.error("Не выбраны методы.")
            return

        run_dir = _make_run_dir(prefix=("file" if source.startswith("Файл") else "synth"))
        st.info(f"Результаты будут сохранены в: {run_dir}")

        with tempfile.TemporaryDirectory() as tmp_dir_obj:
            tmp_dir = str(tmp_dir_obj)

            if source.startswith("Файл"):
                input_path = os.path.join(tmp_dir, uploaded_file.name)
                with open(input_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                shutil.copy2(input_path, run_dir / Path(uploaded_file.name).name)
                dataset_label = uploaded_file.name
            else:
                input_path = os.path.join(tmp_dir, f"{synth_name}.csv")
                assert synth_df is not None
                synth_df.to_csv(input_path, index=False)
                shutil.copy2(input_path, run_dir / f"{synth_name}.csv")
                dataset_label = synth_name

            tool = engine.BigMasterTool()

            with st.spinner("Загрузка данных и расчёт..."):
                tool.load_data_excel(
                    input_path,
                    preprocess=preprocess,
                    normalize=normalize,
                    fill_missing=fill_missing,
                    remove_outliers=remove_outliers,
                    outlier_mode=outlier_mode_key,
                    log_transform=log_transform,
                    detrend_linear=detrend_linear,
                    deseasonalize=deseasonalize,
                    seasonal_period=int(seasonal_period),
                    prewhiten_ar1=prewhiten_ar1,
                )

                method_options: Optional[Dict[str, Any]] = None
                if method_options_text.strip():
                    try:
                        raw = json.loads(method_options_text)
                        if isinstance(raw, dict):
                            method_options = raw
                    except Exception:
                        method_options = None

                w_grid = list(range(int(window_min), int(window_max) + 1, max(1, int(window_step))))
                stride_scan = None if int(window_stride_scan) == 0 else int(window_stride_scan)

                tool.run_selected_methods(
                    selected_methods,
                    max_lag=int(max_lag),
                    lag_selection=lag_selection,
                    lag=int(lag),
                    method_options=method_options,
                    scan_window_pos=bool(scan_window_pos and include_scans),
                    scan_window_size=bool(scan_window_size and include_scans),
                    scan_lag=bool(scan_lag and include_scans),
                    scan_cube=bool(scan_cube and include_scans),
                    window_sizes_grid=w_grid,
                    window_min=int(window_min),
                    window_max=int(window_max),
                    window_step=int(window_step),
                    window_size=int(window_size_default),
                    window_start_min=int(window_start_min),
                    window_start_max=int(window_start_max),
                    window_stride=stride_scan,
                    window_max_windows=int(window_max_windows),
                    lag_min=int(lag_min),
                    lag_max=int(lag_max),
                    lag_step=int(lag_step),
                    cube_combo_limit=int(cube_combo_limit),
                    cube_eval_limit=int(cube_eval_limit),
                    cube_matrix_mode=str(cube_matrix_mode),
                    cube_matrix_limit=int(cube_matrix_limit),
                    cube_gallery_mode=str(cube_gallery_mode),
                    cube_gallery_k=int(cube_gallery_k),
                    cube_gallery_limit=int(cube_gallery_limit),
                )

                series_xlsx = run_dir / "series.xlsx"
                try:
                    with pd.ExcelWriter(series_xlsx) as writer:
                        if getattr(tool, "data_raw", None) is not None and not tool.data_raw.empty:
                            tool.data_raw.to_excel(writer, sheet_name="raw", index=False)
                        if getattr(tool, "data_preprocessed", None) is not None and not tool.data_preprocessed.empty:
                            tool.data_preprocessed.to_excel(writer, sheet_name="preprocessed", index=False)
                        if getattr(tool, "data_after_autodiff", None) is not None and not tool.data_after_autodiff.empty:
                            tool.data_after_autodiff.to_excel(writer, sheet_name="after_autodiff", index=False)
                except Exception:
                    pass

                excel_path = run_dir / "report.xlsx"
                html_path = run_dir / "report.html"

                if output_mode in {"excel", "both"}:
                    tool.export_big_excel(str(excel_path), threshold=threshold, p_value_alpha=alpha)

                if output_mode in {"html", "both"}:
                    tool.export_html_report(
                        str(html_path),
                        graph_threshold=threshold,
                        p_alpha=alpha,
                        include_diagnostics=include_diagnostics,
                        include_scans=include_scans,
                        include_matrix_tables=include_matrix_tables,
                        include_fft_plots=include_fft_plots,
                        harmonic_top_k=int(harmonic_top_k),
                    )

        st.success("Готово.")

        if (run_dir / "report.html").exists():
            st.download_button(
                "Скачать report.html",
                data=(run_dir / "report.html").read_bytes(),
                file_name=f"{dataset_label}_report.html",
                mime="text/html",
            )
        if (run_dir / "report.xlsx").exists():
            st.download_button(
                "Скачать report.xlsx",
                data=(run_dir / "report.xlsx").read_bytes(),
                file_name=f"{dataset_label}_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        if (run_dir / "series.xlsx").exists():
            st.download_button(
                "Скачать series.xlsx (ряды)",
                data=(run_dir / "series.xlsx").read_bytes(),
                file_name=f"{dataset_label}_series.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


if __name__ == "__main__":
    main()
