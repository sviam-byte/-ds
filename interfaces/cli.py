#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Интерфейс командной строки (CLI) для Time Series Analysis Tool.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Добавляем путь к src в sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.engine import BigMasterTool, method_mapping
from src.core.preprocessing import configure_warnings
from src.config import SAVE_FOLDER


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute connectivity measures for multivariate time series.")
    p.add_argument("input_file", help="Path to input CSV or Excel file with time series data")
    p.add_argument("--lags", type=int, default=5, help="Max lag/model order (for Granger, TE, etc.)")
    p.add_argument("--header", choices=["auto", "yes", "no"], default="auto", help="Header mode for input parser")
    p.add_argument("--time-col", dest="time_col", default="auto", help="Time/index column mode: auto|none|<column_name>")
    p.add_argument("--transpose", choices=["auto", "yes", "no"], default="auto", help="Transpose mode for parsed table")
    p.add_argument("--no-preprocess", action="store_true", help="Disable all preprocessing (assume data already prepared)")
    p.add_argument("--log", action="store_true", help="Apply logarithm transform to data (for positive-valued data)")
    p.add_argument("--no-outliers", action="store_true", help="Disable outlier removal")
    p.add_argument("--no-normalize", action="store_true", help="Disable normalization of data")
    p.add_argument("--no-stationarity-check", action="store_true", help="Disable stationarity check (ADF test)")
    p.add_argument("--graph-threshold", type=float, default=0.2, help="Threshold for graph edges (non-pvalue methods)")
    p.add_argument("--p-alpha", type=float, default=0.05, help="Alpha for p-value methods (Granger family)")
    p.add_argument("--output", default=None, help="Output Excel file path")
    p.add_argument("--no-excel", action="store_true", help="Skip Excel export")
    p.add_argument("--no-pairwise", action="store_true", help="Do not include heavy pairwise sheets in Excel")
    p.add_argument("--report-html", default=None, help="Write single-file HTML report to this path")
    p.add_argument("--fs", type=float, default=1.0, help="Sampling rate (Hz) for FFT/coherence/frequency summaries")
    # Diagnostics toggles (HTML)
    p.add_argument("--no-diagnostics", action="store_true", help="Do not include initial analysis block in HTML")
    p.add_argument("--diagnostics-max-series", type=int, default=8, help="Max series to render in diagnostics")
    p.add_argument("--diagnostics-max-pairs", type=int, default=6, help="Max pairs to render for frequency dependence")
    p.add_argument("--no-adf", action="store_true", help="Disable ADF stationarity in diagnostics")
    p.add_argument("--no-hurst", action="store_true", help="Disable Hurst block in diagnostics")
    p.add_argument("--no-seasonality", action="store_true", help="Disable seasonality in diagnostics")
    p.add_argument("--no-fft", action="store_true", help="Disable FFT/power spectrum in diagnostics")
    p.add_argument("--no-ac-ph", action="store_true", help="Disable AC & PH plots in diagnostics")
    p.add_argument("--no-entropy", action="store_true", help="Disable entropy in diagnostics")
    p.add_argument("--no-frequency-summary", action="store_true", help="Disable frequency summary in diagnostics")
    p.add_argument("--no-frequency-dependence", action="store_true", help="Disable frequency dependence in diagnostics")
    p.add_argument("--report-site", default=None, help="Write mini-site report to this directory")
    p.add_argument("--report-site-zip", default=None, help="If set, also zip the site to this path")
    p.add_argument("--quiet-warnings", action="store_true", help="Suppress warnings")
    p.add_argument("--experimental", action="store_true", help="Enable experimental sliding-window analyses")
    return p


def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    configure_warnings(quiet=args.quiet_warnings)

    filepath = os.path.abspath(args.input_file)
    output_path = args.output or os.path.join(SAVE_FOLDER, "AllMethods_Full.xlsx")
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tool = BigMasterTool(enable_experimental=args.experimental)
    try:
        tool.fs = float(args.fs) if args.fs and args.fs > 0 else 1.0
    except Exception:
        tool.fs = 1.0
    tool.lag_ranges = {v: range(1, args.lags + 1) for v in method_mapping}

    tool.load_data_excel(
        filepath,
        header=args.header,
        time_col=args.time_col,
        transpose=args.transpose,
        preprocess=(not args.no_preprocess),
        log_transform=args.log,
        remove_outliers=not args.no_outliers,
        normalize=not args.no_normalize,
        fill_missing=True,
        check_stationarity=not args.no_stationarity_check,
    )
    tool.run_all_methods()
    if not args.no_excel:
        tool.export_big_excel(
            output_path,
            threshold=args.graph_threshold,
            p_value_alpha=args.p_alpha,
            window_size=100,
            overlap=50,
            log_transform=args.log,
            remove_outliers=not args.no_outliers,
            normalize=not args.no_normalize,
            fill_missing=True,
            check_stationarity=not args.no_stationarity_check,
            include_pairwise_sheets=(not args.no_pairwise),
        )
        print("Готово. Excel сохранён в:", output_path)

    if args.report_html:
        tool.export_html_report(
            args.report_html,
            graph_threshold=args.graph_threshold,
            p_alpha=args.p_alpha,
            include_diagnostics=(not args.no_diagnostics),
            diagnostics_max_series=int(args.diagnostics_max_series),
            diagnostics_max_pairs=int(args.diagnostics_max_pairs),
            include_adf=(not args.no_adf),
            include_hurst=(not args.no_hurst),
            include_seasonality=(not args.no_seasonality),
            include_fft=(not args.no_fft),
            include_ac_ph=(not args.no_ac_ph),
            include_entropy=(not args.no_entropy),
            include_frequency_summary=(not args.no_frequency_summary),
            include_frequency_dependence=(not args.no_frequency_dependence),
        )
        print("Готово. HTML отчёт:", os.path.abspath(args.report_html))

    if args.report_site:
        zip_path = args.report_site_zip
        out = tool.export_site_report(
            args.report_site,
            graph_threshold=args.graph_threshold,
            p_alpha=args.p_alpha,
            zip_path=zip_path,
        )
        print("Готово. Site отчёт:", os.path.abspath(out))

    if args.no_excel and not args.report_html and not args.report_site:
        print("Нечего сохранять: включи --report-html и/или --report-site, или убери --no-excel.")


if __name__ == "__main__":
    main()
