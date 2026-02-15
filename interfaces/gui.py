#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Графический интерфейс (Tkinter) с вкладками: Один файл, Пакетная обработка, Генератор.
"""

import os
import sys
import traceback
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pandas as pd

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.engine import BigMasterTool, method_mapping
from src.core import data_loader, generator
from src.config import AnalysisConfig, PYINFORM_AVAILABLE


class App(tk.Tk):
    """Основное GUI-приложение с вкладками для анализа и генерации данных."""

    def __init__(self) -> None:
        super().__init__()
        self.title("TimeSeries Analysis Tool Pro")
        self.geometry("1000x800")

        self.max_lag = tk.IntVar(value=5)
        self.graph_threshold = tk.DoubleVar(value=0.2)
        self.p_alpha = tk.DoubleVar(value=0.05)
        self.auto_diff = tk.BooleanVar(value=False)
        self.p_correction = tk.StringVar(value="none")
        self.method_vars: dict[str, tk.BooleanVar] = {}

        self._init_ui()

    def _init_ui(self) -> None:
        """Инициализирует структуру вкладок и общую панель настроек."""
        tab_control = ttk.Notebook(self)

        self.tab_single = ttk.Frame(tab_control)
        self.tab_batch = ttk.Frame(tab_control)
        self.tab_gen = ttk.Frame(tab_control)

        tab_control.add(self.tab_single, text="Один файл")
        tab_control.add(self.tab_batch, text="Пакетная обработка (Папка)")
        tab_control.add(self.tab_gen, text="Генератор тестов")

        tab_control.pack(expand=1, fill="both")

        self._build_single_tab(self.tab_single)
        self._build_batch_tab(self.tab_batch)
        self._build_gen_tab(self.tab_gen)
        self._build_settings_panel()

    def _build_single_tab(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Анализ одного файла", padding=10)
        frame.pack(fill="x", padx=10, pady=10)

        self.file_path = tk.StringVar()

        row = ttk.Frame(frame)
        row.pack(fill="x", pady=5)
        ttk.Entry(row, textvariable=self.file_path).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Выбрать файл...", command=self._browse_file).pack(side="left", padx=5)

        ttk.Button(frame, text="Запустить анализ (HTML + Excel)", command=self._run_single).pack(pady=10)

    def _build_batch_tab(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Обработка всей папки", padding=10)
        frame.pack(fill="x", padx=10, pady=10)

        self.folder_path = tk.StringVar()

        row = ttk.Frame(frame)
        row.pack(fill="x", pady=5)
        ttk.Entry(row, textvariable=self.folder_path).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Выбрать папку...", command=self._browse_folder).pack(side="left", padx=5)

        self.batch_progress = ttk.Progressbar(frame, orient="horizontal", mode="determinate")
        self.batch_progress.pack(fill="x", pady=10)

        ttk.Button(frame, text="Обработать все файлы в папке", command=self._run_batch).pack(pady=5)
        ttk.Label(
            frame,
            text="* Результаты будут сохранены в подпапке 'analysis_results' внутри выбранной папки",
            foreground="gray",
        ).pack()

    def _build_gen_tab(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Создание синтетических данных", padding=10)
        frame.pack(fill="x", padx=10, pady=10)

        self.gen_samples = tk.IntVar(value=500)
        self.gen_coupling = tk.DoubleVar(value=0.8)

        r1 = ttk.Frame(frame)
        r1.pack(fill="x", pady=5)
        ttk.Label(r1, text="Количество точек:").pack(side="left")
        ttk.Entry(r1, textvariable=self.gen_samples, width=10).pack(side="left", padx=5)

        r2 = ttk.Frame(frame)
        r2.pack(fill="x", pady=5)
        ttk.Label(r2, text="Сила связи (X->Y):").pack(side="left")
        ttk.Entry(r2, textvariable=self.gen_coupling, width=10).pack(side="left", padx=5)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill="x", pady=10)
        ttk.Button(btn_frame, text="Генерировать систему (X->Y, Шум, Сезон)", command=self._run_gen_system).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Генерировать случ. блуждания", command=self._run_gen_rw).pack(side="left", padx=5)

    def _build_settings_panel(self) -> None:
        frame = ttk.LabelFrame(self, text="Настройки методов анализа", padding=10)
        frame.pack(fill="both", expand=True, padx=10, pady=5)

        p_frame = ttk.Frame(frame)
        p_frame.pack(fill="x")
        ttk.Label(p_frame, text="Макс. лаг:").pack(side="left")
        ttk.Spinbox(p_frame, from_=1, to=50, textvariable=self.max_lag, width=5).pack(side="left", padx=5)

        ttk.Label(p_frame, text="Порог графа:").pack(side="left", padx=(15, 0))
        ttk.Entry(p_frame, textvariable=self.graph_threshold, width=5).pack(side="left", padx=5)

        ttk.Checkbutton(p_frame, text="Авто-дифференцирование", variable=self.auto_diff).pack(side="left", padx=15)

        ttk.Label(p_frame, text="Коррекция P-value:").pack(side="left", padx=(15, 5))
        ttk.Combobox(
            p_frame,
            textvariable=self.p_correction,
            values=["none", "fdr_bh"],
            state="readonly",
            width=10,
        ).pack(side="left")

        m_frame = ttk.LabelFrame(frame, text="Методы", padding=5)
        m_frame.pack(fill="both", expand=True, pady=10)

        methods = sorted(method_mapping.keys())
        for i, m in enumerate(methods):
            var = tk.BooleanVar(value=(m in ["correlation_full", "granger_full"]))
            self.method_vars[m] = var
            col = i % 3
            row = i // 3
            name = m
            if not PYINFORM_AVAILABLE and m.startswith("te_"):
                name += " (slow)"
            ttk.Checkbutton(m_frame, text=name, variable=var).grid(row=row, column=col, sticky="w", padx=10)

    def _browse_file(self) -> None:
        f = filedialog.askopenfilename(filetypes=[("Data", "*.csv *.xlsx")])
        if f:
            self.file_path.set(f)

    def _browse_folder(self) -> None:
        d = filedialog.askdirectory()
        if d:
            self.folder_path.set(d)

    def _get_config(self) -> AnalysisConfig:
        return AnalysisConfig(
            max_lag=self.max_lag.get(),
            p_value_alpha=self.p_alpha.get(),
            graph_threshold=self.graph_threshold.get(),
            auto_difference=self.auto_diff.get(),
            pvalue_correction=self.p_correction.get(),
        )

    def _get_selected_methods(self) -> list[str]:
        return [m for m, var in self.method_vars.items() if var.get()]

    def _run_tool(self, df: pd.DataFrame, out_dir: str, name_prefix: str) -> str | None:
        """Запуск движка для одного датафрейма."""
        cfg = self._get_config()
        methods = self._get_selected_methods()

        if not methods:
            messagebox.showwarning("Ошибка", "Выберите хотя бы один метод!")
            return None

        tool = BigMasterTool(df, config=cfg)
        tool.data = df.fillna(df.mean(numeric_only=True))

        if cfg.auto_difference:
            from src.analysis import stats as s_stats

            for c in tool.data.columns:
                if pd.api.types.is_numeric_dtype(tool.data[c]):
                    _, p = s_stats.test_stationarity(tool.data[c])
                    if p is not None and p > 0.05:
                        tool.data[c] = tool.data[c].diff().fillna(0)

        tool.run_selected_methods(methods, max_lag=cfg.max_lag)

        os.makedirs(out_dir, exist_ok=True)
        html_path = os.path.join(out_dir, f"{name_prefix}_report.html")
        excel_path = os.path.join(out_dir, f"{name_prefix}_full.xlsx")

        tool.export_html_report(html_path, graph_threshold=cfg.graph_threshold, p_alpha=cfg.p_value_alpha)
        tool.export_big_excel(excel_path, threshold=cfg.graph_threshold, p_value_alpha=cfg.p_value_alpha)
        return html_path

    def _run_single(self) -> None:
        fp = self.file_path.get()
        if not os.path.exists(fp):
            messagebox.showerror("Error", "Файл не найден")
            return

        try:
            df = data_loader.load_or_generate(fp)
            out_dir = os.path.dirname(fp)
            name = Path(fp).stem

            report = self._run_tool(df, out_dir, name)
            if report and messagebox.askyesno("Готово", "Открыть отчет?"):
                webbrowser.open(report)
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            print(traceback.format_exc())

    def _run_batch(self) -> None:
        folder = self.folder_path.get()
        if not os.path.exists(folder):
            messagebox.showerror("Error", "Папка не найдена")
            return

        files = [f for f in os.listdir(folder) if f.lower().endswith((".csv", ".xlsx"))]
        if not files:
            messagebox.showinfo("Info", "В папке нет файлов данных.")
            return

        out_root = os.path.join(folder, "analysis_results")
        os.makedirs(out_root, exist_ok=True)

        self.batch_progress["maximum"] = len(files)
        self.batch_progress["value"] = 0

        success_count = 0

        for f in files:
            fp = os.path.join(folder, f)
            try:
                df = data_loader.load_or_generate(fp)
                name = Path(f).stem
                file_out_dir = os.path.join(out_root, name)
                self._run_tool(df, file_out_dir, "report")
                success_count += 1
            except Exception as e:
                print(f"Failed to process {f}: {e}")

            self.batch_progress["value"] += 1
            self.update_idletasks()

        messagebox.showinfo("Готово", f"Обработано {success_count} из {len(files)} файлов.\nРезультаты в: {out_root}")

    def _run_gen_system(self) -> None:
        try:
            df = generator.generate_coupled_system(
                n_samples=self.gen_samples.get(),
                coupling_strength=self.gen_coupling.get(),
            )
            self._save_and_open_gen(df, "coupled_system")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _run_gen_rw(self) -> None:
        try:
            df = generator.generate_random_walks(n_samples=self.gen_samples.get())
            self._save_and_open_gen(df, "random_walks")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _save_and_open_gen(self, df: pd.DataFrame, name_base: str) -> None:
        f = filedialog.asksaveasfilename(initialfile=f"{name_base}.csv", filetypes=[("CSV", "*.csv")])
        if f:
            df.to_csv(f, index=False)
            if messagebox.askyesno("Генерация", "Файл сохранен. Открыть его для анализа в первой вкладке?"):
                self.file_path.set(f)


if __name__ == "__main__":
    app = App()
    app.mainloop()
