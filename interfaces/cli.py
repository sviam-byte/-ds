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
    p.add_argument("--pvalue-correction", choices=["none", "fdr_bh"], default="none", help="Multiple testing correction for p-values")
    p.add_argument("--user-config", default="", help="User run config (JSON/dict/key=value;...) for variant presets and tuning")
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

    # Важно для совместимости:
    # - если пользовательский конфиг НЕ задан, оставляем старое поведение run_all_methods().
    # - если конфиг задан, запускаем выборочные варианты с пресетами/тюнингом.
    user_text = (args.user_config or "").strip()
    if args.interactive_config:
        print("\n[Input] Примеры:")
        print("  preset=full")
        print("  variants=mutinf_full,te_directed; max_lag=12; lag_selection=optimize")
        print('  {"preset":"causal","window_sizes":[256,512],"max_lag":12}')
        print("  {'preset':'basic','window_sizes':'256,512','window_policy':'best'}")
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

        tool.run_selected_methods(
            variants,
            max_lag=spec.max_lag,
            lag_selection=spec.lag_selection,
            window_sizes=spec.window_sizes,
            window_stride=spec.window_stride,
            window_policy=spec.window_policy,
            partial_mode=spec.partial_mode,
            pairwise_policy=spec.pairwise_policy,
            custom_controls=spec.custom_controls,
        )
    else:
        tool.load_data_excel(filepath)
        tool.run_all_methods()

    name = Path(filepath).stem
    os.makedirs(out_dir, exist_ok=True)

    excel_path = args.output or os.path.join(out_dir, f"{name}_full.xlsx")
    html_path = args.report_html or os.path.join(out_dir, f"{name}_report.html")

    tool.export_big_excel(excel_path, threshold=args.graph_threshold, p_value_alpha=args.p_alpha)
    tool.export_html_report(html_path, graph_threshold=args.graph_threshold, p_alpha=args.p_alpha)
    print(f"Processed: {filepath}\n  Excel: {os.path.abspath(excel_path)}\n  HTML:  {os.path.abspath(html_path)}")


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
