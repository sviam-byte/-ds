#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Интерфейс командной строки (CLI) для Time Series Analysis Tool."""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import AnalysisConfig, SAVE_FOLDER
from src.core.engine import BigMasterTool
from src.core.variant_presets import expand_variants
from src.io.user_input import build_run_spec, parse_user_input


def build_parser() -> argparse.ArgumentParser:
    """Собирает parser CLI-аргументов."""
    p = argparse.ArgumentParser(description="Compute connectivity measures for multivariate time series.")
    p.add_argument("input_file", nargs="?", default="demo.csv", help="Path to input CSV/Excel file or directory")
    p.add_argument("--lags", type=int, default=5, help="Max lag/model order")
    p.add_argument("--graph-threshold", type=float, default=0.2, help="Threshold for graph edges")
    p.add_argument("--p-alpha", type=float, default=0.05, help="Alpha for p-value methods")
    p.add_argument("--output", default=None, help="Output Excel file path")
    p.add_argument("--report-html", default=None, help="Write single-file HTML report to this path")
    p.add_argument("--experimental", action="store_true", help="Enable experimental analyses")
    p.add_argument("--generate", choices=["coupled", "rw"], help="Generate synthetic data instead of loading file")
    p.add_argument("--output-dir", help="Directory to save results (default: same as input or 'results')")
    p.add_argument("--auto-difference", action="store_true", help="Auto-difference non-stationary series")
    p.add_argument(
        "--pvalue-correction",
        choices=["none", "fdr_bh"],
        default="none",
        help="Multiple testing correction for p-values",
    )
    p.add_argument(
        "--user-config",
        default="",
        help="User run config (JSON/dict/key=value;...) for variant presets and tuning",
    )
    p.add_argument("--interactive-config", action="store_true", help="Read run config interactively from stdin")
    return p


def _process_single_file(filepath: str, args: argparse.Namespace, out_dir: str) -> None:
    """Обрабатывает один файл данных и сохраняет отчеты."""
    cfg = AnalysisConfig(
        max_lag=int(args.lags),
        p_value_alpha=float(args.p_alpha),
        graph_threshold=float(args.graph_threshold),
        enable_experimental=bool(args.experimental),
        auto_difference=bool(args.auto_difference),
        pvalue_correction=args.pvalue_correction,
    )

    tool = BigMasterTool(config=cfg)

    spec = None

    # Важно для совместимости:
    # - если пользовательский конфиг НЕ задан, оставляем старое поведение run_all_methods().
    # - если конфиг задан, запускаем выборочные варианты с пресетами/тюнингом.
    user_text = (args.user_config or "").strip()
    if args.interactive_config:
        print("\n[Input] Примеры:")
        print("  preset=full")
        print("  variants=mutinf_full,te_directed; max_lag=12; lag_selection=optimize")
        print("  variants=mutinf_full,te_directed; lag_selection=fixed; lag=2")
        print("  output_mode=html; include_scans=1; include_fft_plots=0")
        print("  scan_cube=1; window_min=64; window_max=192; window_step=64; lag_min=1; lag_max=3")
        print('  {"preset":"causal","window_sizes":[256,512],"max_lag":12}')
        print(
            '  {"preset":"basic","method_options":{"te_directed":{"scan_cube":0}}}'
        )
        print("Пусто -> дефолты.\n")
        user_text = input("Config> ").strip()

    if user_text:
        user_cfg = parse_user_input(user_text)
        spec = build_run_spec(user_cfg, default_max_lag=int(getattr(tool.config, "max_lag", args.lags)))
        variants, explain = expand_variants(spec.variants)

        print("\n[Plan] Как будет считаться:")
        print(explain)
        print(spec.explain())
        print()

        # Загружаем данные с опциями предобработки из пользовательской спецификации.
        load_kwargs = {"preprocess": bool(spec.preprocess)}
        opts = dict(spec.preprocess_options or {})
        for key in [
            "log_transform",
            "remove_outliers",
            "normalize",
            "fill_missing",
            "check_stationarity",
            "header",
            "time_col",
            "transpose",
        ]:
            if key in opts:
                load_kwargs[key] = opts[key]
        tool.load_data_excel(filepath, **load_kwargs)

        # grids for scans
        if spec.window_sizes_grid:
            w_grid = [int(w) for w in spec.window_sizes_grid if int(w) >= 2]
        else:
            w_grid = list(range(int(spec.window_min), int(spec.window_max) + 1, max(1, int(spec.window_step))))
        if spec.lag_grid:
            l_grid = [int(l) for l in spec.lag_grid if int(l) >= 1]
        else:
            l_grid = None

        tool.run_selected_methods(
            variants,
            max_lag=spec.max_lag,
            lag_selection=spec.lag_selection,
            lag=spec.lag,
            window_sizes=spec.window_sizes,
            window_stride=spec.window_stride,
            window_policy=spec.window_policy,
            partial_mode=spec.partial_mode,
            pairwise_policy=spec.pairwise_policy,
            custom_controls=spec.custom_controls,
            method_options=spec.method_options,
            # legacy: affects main returned matrix
            window_cube_level=spec.window_cube_level,
            window_cube_eval_limit=spec.window_cube_eval_limit,
            # scans
            scan_window_pos=spec.scan_window_pos,
            scan_window_size=spec.scan_window_size,
            scan_lag=spec.scan_lag,
            scan_cube=spec.scan_cube,
            window_sizes_grid=w_grid,
            window_min=spec.window_min,
            window_max=spec.window_max,
            window_step=spec.window_step,
            window_size=spec.window_size,
            window_start_min=spec.window_start_min,
            window_start_max=spec.window_start_max,
            window_max_windows=spec.window_max_windows,
            lag_grid=l_grid,
            lag_min=spec.lag_min,
            lag_max=spec.lag_max,
            lag_step=spec.lag_step,
            cube_combo_limit=spec.cube_combo_limit,
            cube_eval_limit=spec.cube_eval_limit,
            cube_matrix_mode=spec.cube_matrix_mode,
            cube_matrix_limit=spec.cube_matrix_limit,
            cube_gallery_mode=spec.cube_gallery_mode,
            cube_gallery_k=spec.cube_gallery_k,
            cube_gallery_limit=spec.cube_gallery_limit,
        )
    else:
        tool.load_data_excel(filepath)
        tool.run_all_methods()

    name = Path(filepath).stem
    os.makedirs(out_dir, exist_ok=True)

    output_mode = (spec.output_mode if spec else "both")
    do_excel = output_mode in {"excel", "both"}
    do_html = output_mode in {"html", "both"}

    excel_path = args.output or os.path.join(out_dir, f"{name}_full.xlsx")
    html_path = args.report_html or os.path.join(out_dir, f"{name}_report.html")

    # Всегда сохраняем сами ряды отдельным файлом рядом с отчётами
    series_path = os.path.join(out_dir, f"{name}_series.xlsx")
    try:
        tool.export_series_bundle(series_path)
    except Exception:
        pass

    if do_excel:
        tool.export_big_excel(excel_path, threshold=args.graph_threshold, p_value_alpha=args.p_alpha)

    if do_html:
        tool.export_html_report(
            html_path,
            graph_threshold=args.graph_threshold,
            p_alpha=args.p_alpha,
            include_diagnostics=(spec.include_diagnostics if spec else True),
            include_scans=(spec.include_scans if spec else True),
            include_matrix_tables=(spec.include_matrix_tables if spec else True),
            include_fft_plots=(spec.include_fft_plots if spec else False),
            harmonic_top_k=(spec.harmonic_top_k if spec else 5),
            include_series_files=True,
        )

    print(f"Processed: {filepath}")
    if do_excel:
        print(f"  Excel: {os.path.abspath(excel_path)}")
    if do_html:
        print(f"  HTML:  {os.path.abspath(html_path)}")


def main() -> None:
    """Точка входа CLI: одиночный файл, папка или генерация данных."""
    args = build_parser().parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.generate:
        from src.core import generator

        print(f"Generating {args.generate} data...")
        if args.generate == "coupled":
            df = generator.generate_coupled_system()
        else:
            df = generator.generate_random_walks()

        out_name = "synthetic_data.csv"
        df.to_csv(out_name, index=False)
        print(f"Saved to {out_name}")
        args.input_file = out_name

    input_path = os.path.abspath(args.input_file)

    if os.path.isdir(input_path):
        files = glob.glob(os.path.join(input_path, "*.csv")) + glob.glob(os.path.join(input_path, "*.xlsx"))
        print(f"Found {len(files)} files in directory.")

        if not files:
            print("No supported files found.")
            return

        root_out = args.output_dir or os.path.join(input_path, "analysis_results")
        for f in files:
            print(f"Processing {f}...")
            file_out_dir = os.path.join(root_out, Path(f).stem)
            _process_single_file(f, args, file_out_dir)
        return

    out_dir = args.output_dir or os.path.dirname(args.output) if args.output else os.path.join(SAVE_FOLDER, "results")
    _process_single_file(input_path, args, out_dir)


if __name__ == "__main__":
    main()
