#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Графический интерфейс (Tkinter) для Time Series Analysis Tool.
"""

import os
import sys
import traceback
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Добавляем путь к src в sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.engine import BigMasterTool, load_or_generate
from src.config import (
    PYINFORM_AVAILABLE,
    DEFAULT_MAX_LAG,
    DEFAULT_PVALUE_ALPHA,
    DEFAULT_EDGE_THRESHOLD,
)

APP_TITLE = "TimeSeries Local Runner"


METHOD_GROUPS = [
    ("Корреляции", [
        ("correlation_full", "Корреляция (полная)"),
        ("correlation_partial", "Частичная корреляция"),
        ("correlation_directed", "Лаговая корреляция (directed)"),
        ("h2_full", "H² (полная)"),
        ("h2_partial", "H² (partial)"),
        ("h2_directed", "H² (directed, lag)"),
    ]),
    ("Взаимная информация / Энтропии", [
        ("mutinf_full", "Взаимная информация (MI)"),
        ("mutinf_partial", "Частичная MI"),
        ("te_full", "Transfer Entropy (TE)"),
        ("te_partial", "TE (partial)"),
        ("te_directed", "TE (directed)"),
        ("ah_full", "Active information storage (AH)"),
        ("ah_partial", "AH (partial)"),
        ("ah_directed", "AH (directed)"),
    ]),
    ("Когерентность / Granger", [
        ("coherence_full", "Когерентность"),
        ("granger_full", "Granger (p-values)"),
        ("granger_partial", "Granger (partial, p-values)"),
        ("granger_directed", "Granger (directed, p-values)"),
    ]),
]


def _open_file_in_browser(path: Path) -> None:
    try:
        webbrowser.open(path.resolve().as_uri())
    except Exception:
        # fallback
        webbrowser.open("file://" + str(path.resolve()))


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1180x720")

        self.file_path = tk.StringVar(value="")
        self.header_mode = tk.StringVar(value="auto")
        self.transpose_mode = tk.StringVar(value="auto")
        self.time_col = tk.StringVar(value="auto")

        self.preprocess = tk.BooleanVar(value=True)
        self.log_transform = tk.BooleanVar(value=False)
        self.remove_outliers = tk.BooleanVar(value=True)
        self.normalize = tk.BooleanVar(value=True)
        self.adf_check = tk.BooleanVar(value=False)

        self.enable_experimental = tk.BooleanVar(value=False)

        self.max_lag = tk.IntVar(value=int(DEFAULT_MAX_LAG))
        self.lag_step = tk.IntVar(value=1)
        self.pick_best_lag = tk.BooleanVar(value=True)
        self.graph_threshold = tk.DoubleVar(value=float(DEFAULT_EDGE_THRESHOLD))
        self.p_alpha = tk.DoubleVar(value=float(DEFAULT_PVALUE_ALPHA))
        self.embed_images = tk.BooleanVar(value=True)
        self.include_tables = tk.BooleanVar(value=True)

        # sampling rate
        self.fs = tk.DoubleVar(value=1.0)

        # diagnostics
        self.include_diagnostics = tk.BooleanVar(value=True)
        self.diagnostics_max_series = tk.IntVar(value=8)
        self.diagnostics_max_pairs = tk.IntVar(value=6)
        self.include_adf = tk.BooleanVar(value=True)
        self.include_hurst = tk.BooleanVar(value=True)
        self.include_seasonality = tk.BooleanVar(value=True)
        self.include_fft = tk.BooleanVar(value=True)
        self.include_ac_ph = tk.BooleanVar(value=True)
        self.include_entropy = tk.BooleanVar(value=True)
        self.include_frequency_summary = tk.BooleanVar(value=True)
        self.include_frequency_dependence = tk.BooleanVar(value=True)

        # window sweep (для всех методов)
        self.window_min = tk.IntVar(value=200)
        self.window_max = tk.IntVar(value=2000)
        self.window_step = tk.IntVar(value=200)
        self.window_stride = tk.IntVar(value=50)

        # выбор переменных/пар и partial-control
        self.include_vars_text = tk.StringVar(value="all")   # например: all | 1,2,4 | c1,c3
        self.pairs_text = tk.StringVar(value="all")          # all | 1-2,1-3 | c1-c3
        self.partial_mode = tk.StringVar(value="pairwise")   # global | pairwise
        self.pairwise_policy = tk.StringVar(value="others")  # others | custom | none
        self.partial_controls_text = tk.StringVar(value="")  # для global: 3,4 или c3,c4
        self.custom_controls_text = tk.StringVar(value="")   # для pairwise_policy=custom
        self.method_vars = {}  # variant -> BooleanVar
        self._build()

    def _build(self) -> None:
        # Top: file picker
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        entry = ttk.Entry(top, textvariable=self.file_path)
        entry.pack(side="left", fill="x", expand=True)

        ttk.Button(top, text="Browse…", command=self.on_browse).pack(side="left", padx=(8, 0))

        # Parsing + preprocess
        parse = ttk.LabelFrame(self, text="Парсинг и предобработка", padding=10)
        parse.pack(fill="x", padx=10, pady=(0, 10))

        row1 = ttk.Frame(parse)
        row1.pack(fill="x")

        ttk.Label(row1, text="Header:").pack(side="left")
        ttk.Combobox(row1, textvariable=self.header_mode, values=["auto", "yes", "no"], width=10, state="readonly").pack(side="left", padx=(6, 18))

        ttk.Label(row1, text="Transpose:").pack(side="left")
        ttk.Combobox(row1, textvariable=self.transpose_mode, values=["auto", "yes", "no"], width=10, state="readonly").pack(side="left", padx=(6, 18))

        ttk.Label(row1, text="Time col (auto|none|name):").pack(side="left")
        ttk.Entry(row1, textvariable=self.time_col, width=18).pack(side="left", padx=(6, 0))

        row2 = ttk.Frame(parse)
        row2.pack(fill="x", pady=(8, 0))

        ttk.Checkbutton(row2, text="Preprocess", variable=self.preprocess).pack(side="left")
        ttk.Checkbutton(row2, text="Log-transform", variable=self.log_transform).pack(side="left", padx=(12, 0))
        ttk.Checkbutton(row2, text="Remove outliers", variable=self.remove_outliers).pack(side="left", padx=(12, 0))
        ttk.Checkbutton(row2, text="Normalize", variable=self.normalize).pack(side="left", padx=(12, 0))
        ttk.Checkbutton(row2, text="ADF stationarity check", variable=self.adf_check).pack(side="left", padx=(12, 0))

        # Methods
        body = ttk.Frame(self, padding=(10, 0, 10, 10))
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body)
        left.pack(side="left", fill="y")

        methods_box = ttk.LabelFrame(left, text="Методы (что считать)", padding=10)
        methods_box.pack(fill="both", expand=True)

        for group_name, items in METHOD_GROUPS:
            lf = ttk.LabelFrame(methods_box, text=group_name, padding=8)
            lf.pack(fill="x", pady=(0, 8))
            for variant, title in items:
                v = tk.BooleanVar(value=False)
                label = title
                if (not PYINFORM_AVAILABLE) and variant.startswith("te_"):
                    label = title + " (fallback)"
                cb = ttk.Checkbutton(lf, text=label, variable=v)
                cb.pack(anchor="w")
                self.method_vars[variant] = v

        # right: params + run
        right = ttk.Frame(body, padding=(10, 0, 0, 0))
        right.pack(side="left", fill="both", expand=True)

        params = ttk.LabelFrame(right, text="Параметры отчёта", padding=10)
        params.pack(fill="x")

        r1 = ttk.Frame(params)
        r1.pack(fill="x")
        ttk.Label(r1, text="Max lag:").pack(side="left")
        ttk.Spinbox(r1, from_=1, to=200, textvariable=self.max_lag, width=8).pack(side="left", padx=(6, 18))

        ttk.Label(r1, text="Lag step:").pack(side="left")
        ttk.Spinbox(r1, from_=1, to=50, textvariable=self.lag_step, width=6).pack(side="left", padx=(6, 18))

        ttk.Label(r1, text="Graph threshold:").pack(side="left")
        ttk.Entry(r1, textvariable=self.graph_threshold, width=8).pack(side="left", padx=(6, 18))

        ttk.Label(r1, text="Sampling rate fs (Hz):").pack(side="left")
        ttk.Entry(r1, textvariable=self.fs, width=8).pack(side="left", padx=(6, 18))

        ttk.Checkbutton(r1, text="Pick best lag (для lag-методов)", variable=self.pick_best_lag).pack(side="left")

        r2 = ttk.Frame(params)
        r2.pack(fill="x", pady=(8, 0))

        ttk.Label(r2, text="p-alpha (pvalue methods):").pack(side="left")
        ttk.Entry(r2, textvariable=self.p_alpha, width=8).pack(side="left", padx=(6, 18))

        ttk.Checkbutton(r2, text="Embed images (one-file HTML)", variable=self.embed_images).pack(side="left")
        ttk.Checkbutton(r2, text="Include matrix tables", variable=self.include_tables).pack(side="left", padx=(12, 0))

        r2fs = ttk.Frame(params)
        r2fs.pack(fill="x", pady=(8, 0))
        ttk.Label(r2fs, text="Sampling rate fs (Hz):").pack(side="left")
        ttk.Entry(r2fs, textvariable=self.fs, width=10).pack(side="left", padx=(6, 18))

        r2b = ttk.Frame(params)
        r2b.pack(fill="x", pady=(8, 0))
        ttk.Checkbutton(r2b, text="Include initial diagnostics", variable=self.include_diagnostics).pack(side="left")
        ttk.Label(r2b, text="Max series").pack(side="left", padx=(12, 4))
        ttk.Entry(r2b, textvariable=self.diagnostics_max_series, width=6).pack(side="left")
        ttk.Label(r2b, text="Max pairs").pack(side="left", padx=(12, 4))
        ttk.Entry(r2b, textvariable=self.diagnostics_max_pairs, width=6).pack(side="left")

        r2c = ttk.Frame(params)
        r2c.pack(fill="x", pady=(6, 0))
        ttk.Checkbutton(r2c, text="ADF", variable=self.include_adf).pack(side="left")
        ttk.Checkbutton(r2c, text="Hurst", variable=self.include_hurst).pack(side="left", padx=(10,0))
        ttk.Checkbutton(r2c, text="Seasonality", variable=self.include_seasonality).pack(side="left", padx=(10,0))
        ttk.Checkbutton(r2c, text="Entropy", variable=self.include_entropy).pack(side="left", padx=(10,0))

        r2d = ttk.Frame(params)
        r2d.pack(fill="x", pady=(6, 0))
        ttk.Checkbutton(r2d, text="FFT", variable=self.include_fft).pack(side="left")
        ttk.Checkbutton(r2d, text="AC & PH", variable=self.include_ac_ph).pack(side="left", padx=(10,0))
        ttk.Checkbutton(r2d, text="Freq summary", variable=self.include_frequency_summary).pack(side="left", padx=(10,0))
        ttk.Checkbutton(r2d, text="Freq dependence", variable=self.include_frequency_dependence).pack(side="left", padx=(10,0))

        r3 = ttk.Frame(params)
        r3.pack(fill="x", pady=(8, 0))
        ttk.Checkbutton(r3, text="Enable experimental (TE/AH)", variable=self.enable_experimental).pack(side="left")

        r4 = ttk.Frame(params)
        r4.pack(fill="x", pady=(10, 0))
        ttk.Label(r4, text="Window min:").pack(side="left")
        ttk.Spinbox(r4, from_=10, to=1000000, textvariable=self.window_min, width=8).pack(side="left", padx=(6, 18))
        ttk.Label(r4, text="Window max:").pack(side="left")
        ttk.Spinbox(r4, from_=10, to=1000000, textvariable=self.window_max, width=8).pack(side="left", padx=(6, 18))
        ttk.Label(r4, text="Window step:").pack(side="left")
        ttk.Spinbox(r4, from_=1, to=1000000, textvariable=self.window_step, width=8).pack(side="left", padx=(6, 18))
        ttk.Label(r4, text="Window stride:").pack(side="left")
        ttk.Spinbox(r4, from_=1, to=1000000, textvariable=self.window_stride, width=8).pack(side="left", padx=(6, 0))


        r5 = ttk.Frame(params)
        r5.pack(fill="x", pady=(10, 0))
        ttk.Label(r5, text="Vars (all | 1,2,4 | c1,c3):").pack(side="left")
        ttk.Entry(r5, textvariable=self.include_vars_text, width=26).pack(side="left", padx=(6, 18))
        ttk.Label(r5, text="Pairs (all | 1-2,1-3 | c1-c3):").pack(side="left")
        ttk.Entry(r5, textvariable=self.pairs_text, width=26).pack(side="left", padx=(6, 0))

        r6 = ttk.Frame(params)
        r6.pack(fill="x", pady=(10, 0))
        ttk.Label(r6, text="Partial mode:").pack(side="left")
        ttk.OptionMenu(r6, self.partial_mode, self.partial_mode.get(), "pairwise", "global").pack(side="left", padx=(6, 18))
        ttk.Label(r6, text="Pairwise policy:").pack(side="left")
        ttk.OptionMenu(r6, self.pairwise_policy, self.pairwise_policy.get(), "others", "custom", "none").pack(side="left", padx=(6, 18))
        ttk.Label(r6, text="Global controls (для global):").pack(side="left")
        ttk.Entry(r6, textvariable=self.partial_controls_text, width=16).pack(side="left", padx=(6, 18))
        ttk.Label(r6, text="Custom controls (для custom):").pack(side="left")
        ttk.Entry(r6, textvariable=self.custom_controls_text, width=16).pack(side="left", padx=(6, 0))
        run_row = ttk.Frame(right, padding=(0, 14, 0, 0))
        run_row.pack(fill="x")
        ttk.Button(run_row, text="Run → HTML", command=self.on_run).pack(side="left")

        ttk.Button(run_row, text="Select none", command=self.on_select_none).pack(side="right")
        ttk.Button(run_row, text="Select all", command=self.on_select_all).pack(side="right", padx=(0, 8))

        self.status = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.status, foreground="gray").pack(anchor="w", pady=(10, 0))

    def on_browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Выбери файл",
            filetypes=[("Data files", "*.csv *.xlsx *.xls"), ("CSV", "*.csv"), ("Excel", "*.xlsx *.xls"), ("All files", "*.*")],
        )
        if path:
            self.file_path.set(path)

    def on_select_all(self) -> None:
        for v in self.method_vars.values():
            try:
                v.set(True)
            except Exception:
                pass

    def on_select_none(self) -> None:
        for v in self.method_vars.values():
            try:
                v.set(False)
            except Exception:
                pass


    def _parse_vars(self, text: str, ncols: int) -> list:
        t = (text or "").strip().lower()
        if not t or t == "all":
            return [f"c{i+1}" for i in range(ncols)]
        # допускаем 'c1,c3' или '1,3'
        parts = [p.strip() for p in t.replace(";", ",").split(",") if p.strip()]
        out = []
        for p in parts:
            if p.startswith("c"):
                try:
                    k = int(p[1:])
                    if 1 <= k <= ncols:
                        out.append(f"c{k}")
                except Exception:
                    pass
            else:
                try:
                    k = int(p)
                    if 1 <= k <= ncols:
                        out.append(f"c{k}")
                except Exception:
                    pass
        return sorted(set(out), key=lambda s: int(s[1:]))

    def _parse_pairs(self, text: str, ncols: int) -> list:
        t = (text or "").strip().lower()
        if not t or t == "all":
            cols = [f"c{i+1}" for i in range(ncols)]
            return [(cols[i], cols[j]) for i in range(ncols) for j in range(i+1, ncols)]
        parts = [p.strip() for p in t.replace(";", ",").split(",") if p.strip()]
        pairs = []
        for p in parts:
            p = p.replace(" ", "")
            if "-" not in p:
                continue
            a, b = p.split("-", 1)
            def norm(x):
                if x.startswith("c"):
                    return x
                return f"c{x}"
            a, b = norm(a), norm(b)
            try:
                ia, ib = int(a[1:]), int(b[1:])
                if 1 <= ia <= ncols and 1 <= ib <= ncols and ia != ib:
                    x, y = (a, b) if ia < ib else (b, a)
                    pairs.append((x, y))
            except Exception:
                pass
        # unique
        seen = set()
        out = []
        for p in pairs:
            if p not in seen:
                out.append(p)
                seen.add(p)
        return out

    def _parse_controls(self, text: str, ncols: int) -> list:
        return self._parse_vars(text, ncols)

    def _get_selected_methods(self):
        return [k for k, v in self.method_vars.items() if bool(v.get())]

    def on_run(self) -> None:
        fp = self.file_path.get().strip()
        if not fp or not os.path.exists(fp):
            messagebox.showerror("Ошибка", "Выбери существующий файл (CSV/XLSX).")
            return

        variants = self._get_selected_methods()
        if not variants:
            messagebox.showerror("Ошибка", "Не выбраны методы для расчёта.")
            return

        self.status.set("Running…")
        self.update_idletasks()

        try:
            df = load_or_generate(
                fp,
                header=self.header_mode.get(),
                time_col=self.time_col.get(),
                transpose=self.transpose_mode.get(),
                preprocess=bool(self.preprocess.get()),
                log_transform=bool(self.log_transform.get()),
                remove_outliers=bool(self.remove_outliers.get()),
                normalize=bool(self.normalize.get()),
                check_stationarity=bool(self.adf_check.get()),
            )

            tool = BigMasterTool(df, enable_experimental=bool(self.enable_experimental.get()))
            try:
                fs = float(self.fs.get())
                tool.fs = fs if fs > 0 else 1.0
            except Exception:
                tool.fs = 1.0
            ncols = int(tool.data.shape[1])
            include_vars = self._parse_vars(self.include_vars_text.get(), ncols)
            pair_filter = self._parse_pairs(self.pairs_text.get(), ncols)
            partial_controls = self._parse_controls(self.partial_controls_text.get(), ncols)
            custom_controls = self._parse_controls(self.custom_controls_text.get(), ncols)
            tool.run_selected_methods(
                variants,
                max_lag=int(self.max_lag.get()),
                lag_step=int(self.lag_step.get()),
                pick_best_lag=bool(self.pick_best_lag.get()),
                window_min=int(self.window_min.get()),
                window_max=int(self.window_max.get()),
                window_step=int(self.window_step.get()),
                window_stride=int(self.window_stride.get()),
                include_vars=include_vars,
                pair_filter=pair_filter,
                partial_mode=str(self.partial_mode.get()),
                pairwise_policy=str(self.pairwise_policy.get()),
                partial_controls=partial_controls,
                custom_controls=custom_controls,
            )

            out_dir = Path(__file__).resolve().parent / "TimeSeriesAnalysis"
            out_dir.mkdir(exist_ok=True)
            out_path = out_dir / f"report_{Path(fp).stem}.html"

            tool.export_html_report(
                str(out_path),
                variants=variants,
                graph_threshold=float(self.graph_threshold.get()),
                p_alpha=float(self.p_alpha.get()),
                embed_images=bool(self.embed_images.get()),
                include_matrix_tables=bool(self.include_tables.get()),
                include_diagnostics=bool(self.include_diagnostics.get()),
                diagnostics_max_series=int(self.diagnostics_max_series.get()),
                diagnostics_max_pairs=int(self.diagnostics_max_pairs.get()),
                include_adf=bool(self.include_adf.get()),
                include_hurst=bool(self.include_hurst.get()),
                include_seasonality=bool(self.include_seasonality.get()),
                include_fft=bool(self.include_fft.get()),
                include_ac_ph=bool(self.include_ac_ph.get()),
                include_entropy=bool(self.include_entropy.get()),
                include_frequency_summary=bool(self.include_frequency_summary.get()),
                include_frequency_dependence=bool(self.include_frequency_dependence.get()),
            )

            self.status.set(f"Saved: {out_path}")
            _open_file_in_browser(out_path)
        except Exception as e:
            tb = traceback.format_exc()
            self.status.set("Error")
            messagebox.showerror("Ошибка", f"{e}\n\n{tb}")
        finally:
            self.status.set("")

def main() -> None:
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
