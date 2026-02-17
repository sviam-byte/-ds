"""Веб-интерфейс (Streamlit) для Time Series Analysis Tool (локально)."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import EXPERIMENTAL_METHODS, SAVE_FOLDER, STABLE_METHODS
from src.core import engine, generator


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




def _make_run_dir(stem: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(ch for ch in (stem or "run") if ch.isalnum() or ch in "-_ ").strip().replace(" ", "_")
    run_dir = Path(SAVE_FOLDER) / "runs" / f"{safe}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _maybe_reset_formula_defaults(preset: str) -> None:
    defaults = {
        "Custom": {"x": "sin(2*pi*t/50) + 0.2*randn()", "y": "0.8*X + 0.3*randn()", "z": "rw(0.5)"},
        "Random": {"x": "randn()", "y": "randn()", "z": "randn()"},
        "Linear + noise": {"x": "0.01*t + 0.2*randn()", "y": "0.5*X + 0.2*randn()", "z": "-0.005*t + 0.2*randn()"},
        "Sin/Cos coupling": {"x": "sin(2*pi*t/50) + 0.1*randn()", "y": "cos(2*pi*t/50) + 0.4*X + 0.1*randn()", "z": "sin(2*pi*t/15) + 0.1*randn()"},
        "AR(1)": {"x": "ar1(phi=0.85, scale=0.5)", "y": "0.7*X + ar1(phi=0.6, scale=0.4)", "z": "ar1(phi=0.3, scale=0.8)"},
    }
    if st.session_state.get("_preset_prev") != preset:
        st.session_state["_preset_prev"] = preset
        d = defaults.get(preset, defaults["Custom"])
        st.session_state["x_expr"] = d["x"]
        st.session_state["y_expr"] = d["y"]
        st.session_state["z_expr"] = d["z"]

def main() -> None:
    st.set_page_config(page_title="Анализ Временных Рядов (Локально)", layout="wide")
    st.title("Анализ Связности Временных Рядов")
    st.caption(f"Локальная версия. Результаты сохраняются в папку: {SAVE_FOLDER}")

    source = st.radio(
        "Источник данных",
        ["Файл (CSV/XLSX)", "Синтетика (формулы)"],
        index=0,
        horizontal=True,
    )

    uploaded_file = None
    synth_df: pd.DataFrame | None = None
    synth_name = "synthetic"

    if source.startswith("Файл"):
        uploaded_file = st.file_uploader("Выберите файл", type=["csv", "xlsx"])
    else:
        with st.expander("Синтетика: формулы X/Y/Z", expanded=True):
            c0, c1, c2 = st.columns(3)
            with c0:
                preset = st.selectbox(
                    "Шаблон",
                    ["Custom", "Random", "Linear + noise", "Sin/Cos coupling", "AR(1)"],
                    index=2,
                )
                _maybe_reset_formula_defaults(preset)
            with c1:
                n_samples = st.number_input("n_samples", min_value=20, max_value=200000, value=800, step=10, key="n_samples")
                dt = st.number_input("dt", min_value=0.0001, max_value=1000.0, value=1.0, step=0.1, format="%.4f", key="dt")
            with c2:
                seed = st.number_input("seed", min_value=0, max_value=10_000_000, value=42, step=1, key="seed")

            st.caption(
                "Переменные: t (время), X (первый ряд), Y (второй), Z (третий). Функции: sin, cos, exp, log, sqrt, "
                "randn(scale=1), rw(scale=1), ar1(phi=0.7, scale=1)."
            )

            x_expr = st.text_input("X(t) =", key="x_expr")
            y_expr = st.text_input("Y(t, X) =", key="y_expr")
            z_expr = st.text_input("Z(t, X, Y) =", key="z_expr")

            synth_name = st.text_input("Имя набора (для папки/файлов)", value=synth_name)

            if st.button("Сгенерировать preview", type="secondary"):
                try:
                    synth_df = generator.generate_formula_dataset(
                        n_samples=int(n_samples),
                        dt=float(dt),
                        seed=int(seed),
                        specs=[
                            generator.FormulaSpec("X", x_expr),
                            generator.FormulaSpec("Y", y_expr),
                            generator.FormulaSpec("Z", z_expr),
                        ],
                    )
                    st.success(f"OK: shape={synth_df.shape}")
                    with st.expander("Preview рядов", expanded=False):
                        st.line_chart(synth_df)
                        st.dataframe(synth_df.head(200))
                except Exception as e:
                    st.error(f"Ошибка генерации: {e}")

    with st.expander("Параметры запуска", expanded=True):
        colA, colB, colC = st.columns(3)
        with colA:
            lag_selection = st.selectbox("Выбор лага (основной расчёт)", ["optimize", "fixed"], index=0)
            if lag_selection == "fixed":
                lag = st.number_input("lag (если fixed)", min_value=1, max_value=200, value=1)
                max_lag = st.number_input("max_lag (для сканов/ограничений)", min_value=1, max_value=200, value=12)
            else:
                max_lag = st.number_input("max_lag (для optimize)", min_value=1, max_value=200, value=12)
                lag = st.number_input("lag (не используется при optimize)", min_value=1, max_value=200, value=1)

            alpha = st.number_input("P-value alpha (для Granger/p-value)", 0.0001, 0.5, 0.05, format="%.4f")
            threshold = st.number_input("Порог графа (Threshold)", 0.0, 1.0, 0.2, 0.05)

        with colB:
            normalize = st.checkbox("Нормализация (Z-score)", value=True)
            preprocess = st.checkbox("Предобработка (fill/outliers/log)", value=True)
            fill_missing = st.checkbox("Заполнять пропуски (interp)", value=True)
            remove_outliers = st.checkbox("Убирать выбросы (Z)", value=True)
            log_transform = st.checkbox("Лог-преобразование (только >0)", value=False)

        with colC:
            output_mode = st.selectbox("Режим вывода", ["both", "html", "excel"], index=0)
            include_diagnostics = st.checkbox("HTML: показывать диагностику", value=True)
            include_scans = st.checkbox("HTML: показывать сканы (окна/лаги/куб)", value=True)
            include_matrix_tables = st.checkbox("HTML: показывать таблицу матрицы (текстом)", value=False)
            include_fft_plots = st.checkbox("HTML: FFT-графики", value=True)
            harmonic_top_k = st.number_input("Гармоники: top_k", min_value=1, max_value=20, value=5)

        st.markdown("---")
        st.subheader("Основной расчёт (что вернётся как итоговая матрица)")
        c1, c2, c3 = st.columns(3)
        with c1:
            use_main_windows = st.checkbox("Использовать окна в основном расчёте", value=False)
            window_policy = st.selectbox("Политика окон (main)", ["best", "mean"], index=0)
            window_stride_main = st.number_input("stride (main, 0=auto)", min_value=0, max_value=100000, value=0, step=1)
        with c2:
            window_sizes_text = st.text_input("main window_sizes", value="256,512")
            st.caption("Если выключено 'использовать окна' — будет считаться на полном интервале.")
        with c3:
            window_cube_level = st.selectbox("Main window×lag×position (legacy)", ["off", "basic", "full"], index=0)
            window_cube_eval_limit = st.number_input("Main-cube eval_limit", min_value=20, max_value=5000, value=120, step=10)

        st.markdown("---")
        st.subheader("Сканы (отчёт/инспектор; не меняют итоговую матрицу)")
        s1, s2, s3 = st.columns(3)
        with s1:
            scan_window_pos = st.checkbox("scan_window_pos", value=True, disabled=not include_scans)
            scan_window_size = st.checkbox("scan_window_size", value=True, disabled=not include_scans)
            scan_lag = st.checkbox("scan_lag", value=True, disabled=not include_scans)
            scan_cube = st.checkbox("scan_cube", value=True, disabled=not include_scans)

        with s2:
            window_min = st.number_input("window_min", min_value=2, max_value=1000000, value=64, step=1, disabled=not include_scans)
            window_max = st.number_input("window_max", min_value=2, max_value=1000000, value=192, step=1, disabled=not include_scans)
            window_step = st.number_input("window_step", min_value=1, max_value=1000000, value=64, step=1, disabled=not include_scans)
            window_size_default = st.number_input("window_size (для scan_window_pos)", min_value=2, max_value=1000000, value=128, step=1, disabled=not include_scans)

        with s3:
            window_start_min = st.number_input("window_start_min (0=auto)", min_value=0, max_value=10_000_000, value=0, step=1, disabled=not include_scans)
            window_start_max = st.number_input("window_start_max (0=auto)", min_value=0, max_value=10_000_000, value=0, step=1, disabled=not include_scans)
            window_stride_scan = st.number_input("window_stride (scan, 0=auto)", min_value=0, max_value=10_000_000, value=0, step=1, disabled=not include_scans)
            window_max_windows = st.number_input("window_max_windows", min_value=1, max_value=5000, value=60, step=1, disabled=not include_scans)

        st.markdown("**Лаг-сетка (для scan_lag и cube)**")
        l1, l2, l3 = st.columns(3)
        with l1:
            lag_min = st.number_input("lag_min", min_value=1, max_value=2000, value=1, step=1, disabled=not include_scans)
        with l2:
            lag_max = st.number_input("lag_max", min_value=1, max_value=2000, value=min(3, int(max_lag)), step=1, disabled=not include_scans)
        with l3:
            lag_step = st.number_input("lag_step", min_value=1, max_value=2000, value=1, step=1, disabled=not include_scans)

        st.markdown("**Куб (window_size × lag × position)**")
        k1, k2, k3 = st.columns(3)
        with k1:
            cube_combo_limit = st.number_input("cube_combo_limit (по парам w×lag)", min_value=1, max_value=200000, value=9, step=1, disabled=not include_scans)
            cube_eval_limit = st.number_input("cube_eval_limit (общий лимит точек)", min_value=1, max_value=2_000_000, value=225, step=5, disabled=not include_scans)
        with k2:
            cube_matrix_mode = st.selectbox("cube_matrix_mode", ["all", "selected"], index=0, disabled=not include_scans)
            cube_matrix_limit = st.number_input("cube_matrix_limit", min_value=1, max_value=2_000_000, value=225, step=5, disabled=not include_scans)
        with k3:
            cube_gallery_mode = st.selectbox("cube_gallery_mode", ["extremes", "topbottom", "quantiles"], index=0, disabled=not include_scans)
            cube_gallery_k = st.number_input("cube_gallery_k", min_value=1, max_value=1000, value=1, step=1, disabled=not include_scans)
            cube_gallery_limit = st.number_input("cube_gallery_limit", min_value=3, max_value=5000, value=60, step=5, disabled=not include_scans)

        st.markdown("---")
        st.subheader("Метод-специфичные оверрайды (advanced)")
        method_options_text = st.text_area(
            "method_options (JSON, ключ = метод)",
            value="",
            placeholder='Напр.: {"te_directed": {"scan_cube": false, "cube_matrix_mode": "selected"}}',
            height=80,
        )

    all_methods = STABLE_METHODS + EXPERIMENTAL_METHODS
    selected_methods = st.multiselect("Выберите методы", all_methods, default=STABLE_METHODS[:2])

    if st.button("Запустить анализ", type="primary"):
        if source.startswith("Файл") and not uploaded_file:
            st.error("Файл не загружен!")
            return

        # Готовим run-dir
        stem = (Path(uploaded_file.name).stem if uploaded_file else synth_name) or "run"
        run_dir = _make_run_dir(stem)

        # Сохраняем входные данные (или синтетические)
        input_path: Path
        try:
            if uploaded_file:
                input_path = run_dir / uploaded_file.name
                input_path.write_bytes(uploaded_file.getbuffer())
            else:
                # генерируем синтетику прямо сейчас (без отдельного клика)
                x_expr = st.session_state.get("x_expr", "sin(2*pi*t/50) + 0.2*randn()")
                y_expr = st.session_state.get("y_expr", "0.8*X + 0.3*randn()")
                z_expr = st.session_state.get("z_expr", "rw(0.5)")
                # дефолты, если preview не запускали
                n_samples = int(st.session_state.get("n_samples", 800) or 800)
                dt = float(st.session_state.get("dt", 1.0) or 1.0)
                seed = int(st.session_state.get("seed", 42) or 42)

                synth_df = generator.generate_formula_dataset(
                    n_samples=n_samples,
                    dt=dt,
                    seed=seed,
                    specs=[
                        generator.FormulaSpec("X", x_expr),
                        generator.FormulaSpec("Y", y_expr),
                        generator.FormulaSpec("Z", z_expr),
                    ],
                )
                input_path = run_dir / f"{stem}_input.csv"
                synth_df.to_csv(input_path, index=False)
        except Exception as e:
            st.error(f"Ошибка подготовки данных: {e}")
            return

        tool = engine.BigMasterTool()

        with st.spinner("Загрузка и расчёт..."):
            try:
                tool.load_data_excel(
                    str(input_path),
                    preprocess=preprocess,
                    normalize=normalize,
                    fill_missing=fill_missing,
                    remove_outliers=remove_outliers,
                    log_transform=log_transform,
                )

                # main windows
                window_sizes_main = None
                if use_main_windows:
                    window_sizes_main = _parse_int_list_text(window_sizes_text)

                # scans/main используют общий параметр window_stride в движке.
                stride_scan = None if int(window_stride_scan) == 0 else int(window_stride_scan)
                stride_main = None if int(window_stride_main) == 0 else int(window_stride_main)
                run_window_stride = stride_scan if stride_scan is not None else stride_main

                # method options
                method_options = None
                if method_options_text.strip():
                    try:
                        method_options = json.loads(method_options_text)
                        if not isinstance(method_options, dict):
                            method_options = None
                    except Exception:
                        method_options = None

                w_grid = list(range(int(window_min), int(window_max) + 1, max(1, int(window_step))))

                tool.run_selected_methods(
                    selected_methods,
                    max_lag=int(max_lag),
                    lag_selection=lag_selection,
                    lag=int(lag),
                    window_sizes=window_sizes_main,
                    window_stride=run_window_stride,
                    window_policy=window_policy,
                    window_cube_level=window_cube_level,
                    window_cube_eval_limit=int(window_cube_eval_limit),
                    method_options=method_options,
                    # scans
                    scan_window_pos=(bool(scan_window_pos) if include_scans else False),
                    scan_window_size=(bool(scan_window_size) if include_scans else False),
                    scan_lag=(bool(scan_lag) if include_scans else False),
                    scan_cube=(bool(scan_cube) if include_scans else False),
                    window_sizes_grid=w_grid,
                    window_min=int(window_min),
                    window_max=int(window_max),
                    window_step=int(window_step),
                    window_size=int(window_size_default),
                    window_start_min=int(window_start_min),
                    window_start_max=int(window_start_max),
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

                # Всегда сохраняем ряды отдельным файлом рядом с отчётами.
                series_path = run_dir / f"{stem}_series.xlsx"
                try:
                    tool.export_series_bundle(str(series_path))
                except Exception:
                    pass

                excel_path = run_dir / f"{stem}_full.xlsx"
                html_path = run_dir / f"{stem}_report.html"

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
                        include_series_files=True,
                    )

                st.success("Готово!")
                st.code(str(run_dir))

                c1, c2, c3 = st.columns(3)
                with c1:
                    if output_mode in {"excel", "both"} and excel_path.exists():
                        st.download_button("Скачать Excel", excel_path.read_bytes(), excel_path.name)
                with c2:
                    if output_mode in {"html", "both"} and html_path.exists():
                        st.download_button("Скачать HTML", html_path.read_bytes(), html_path.name)
                with c3:
                    if series_path.exists():
                        st.download_button("Скачать ряды (xlsx)", series_path.read_bytes(), series_path.name)

                # Ряды раскрываются только по клику
                with st.expander("Исходные ряды (preview)", expanded=False):
                    try:
                        df_show = tool.data_raw if not tool.data_raw.empty else tool.data
                        st.line_chart(df_show)
                        st.dataframe(df_show.head(200))
                    except Exception:
                        pass

                st.subheader("Предварительный просмотр матриц")
                from src.visualization import plots

                for method in selected_methods:
                    mat = tool.results.get(method)
                    if mat is None:
                        continue
                    chosen = None
                    try:
                        chosen = (tool.results_meta.get(method) or {}).get("chosen_lag")
                    except Exception:
                        chosen = None
                    title = f"{method}" + (f" (chosen_lag={chosen})" if chosen is not None else "")
                    buf = plots.plot_heatmap(mat, title)
                    st.image(buf, caption=title)

            except Exception as e:
                st.error(f"Ошибка выполнения: {e}")
                import traceback

                st.text(traceback.format_exc())


if __name__ == "__main__":
    main()
