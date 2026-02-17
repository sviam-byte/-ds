"""
Генератор HTML-отчетов для BigMasterTool.
"""

from __future__ import annotations

import base64
import html
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.config import METHOD_INFO, is_directed_method, is_pvalue_method
from src.visualization import plots


@dataclass(slots=True)
class HTMLReportGenerator:
    """Генерирует HTML-отчет, используя данные и результаты анализа."""

    tool: object  # Ссылка на экземпляр BigMasterTool

    def _b64_png(self, buf: BytesIO) -> str:
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _plot_matrix_b64(self, mat: np.ndarray, title: str, cols: list) -> str:
        buf = plots.plot_heatmap(mat, title, labels=cols)
        return self._b64_png(buf)

    def _plot_curve_b64(self, xs, ys, title: str, xlab: str) -> str:
        buf = BytesIO()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(xs, ys, marker="o")
        ax.set_title(title)
        ax.set_xlabel(xlab)
        ax.set_ylabel("Качество (метрика)")
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return self._b64_png(buf)

    def _plot_fft_b64(self, series, title: str, fs: float = 1.0) -> str:
        """Возвращает base64 PNG с FFT-спектром для одного ряда."""
        return self._b64_png(plots.plot_fft_spectrum(series, title, fs=fs))

    def _plot_cube3d_b64(self, points, title: str) -> str:
        """Возвращает base64 PNG с 3D-графиком window×lag×position."""
        return self._b64_png(plots.plot_window_cube_3d(points, title))


    def _jsonable_matrix(self, mat) -> list | None:
        """Переводит матрицу в JSON-friendly list[list[float|None]]."""
        if mat is None:
            return None
        try:
            arr = np.asarray(mat, dtype=float)
        except Exception:
            return None
        if arr.ndim != 2 or arr.size == 0:
            return None
        out: list[list] = []
        for i in range(arr.shape[0]):
            row = []
            for j in range(arr.shape[1]):
                v = float(arr[i, j])
                row.append(v if np.isfinite(v) else None)
            out.append(row)
        return out

    def _build_scan_payload(self, *, variant: str, anchor: str, cols: list, sc: dict) -> dict:
        """Собирает компактный JSON payload для интерактивных сканов в HTML."""
        payload: dict = {"variant": variant, "anchor": anchor, "labels": list(cols)}

        pos = sc.get("window_pos") or {}
        if isinstance(pos, dict) and (pos.get("curve") or pos.get("ticks")):
            mats = {}
            ticks = []
            for t in (pos.get("ticks") or []):
                tid = t.get("id")
                if not tid:
                    continue
                ticks.append({"id": tid, "start": t.get("start"), "end": t.get("end"), "metric": t.get("metric")})
                m = self._jsonable_matrix(t.get("matrix"))
                if m is not None:
                    mats[str(tid)] = m
            payload["window_pos"] = {
                "window_size": pos.get("window_size"),
                "stride": pos.get("stride"),
                "lag": pos.get("lag"),
                "curve": pos.get("curve") or {},
                "ticks": ticks,
                "extremes": pos.get("extremes") or {},
                "matrices": mats,
            }

        ws = sc.get("window_size") or {}
        if isinstance(ws, dict) and (ws.get("curve") or ws.get("ticks")):
            mats = {}
            ticks = []
            for t in (ws.get("ticks") or []):
                tid = t.get("id")
                if not tid:
                    continue
                ticks.append({
                    "id": tid,
                    "window_size": t.get("window_size"),
                    "start": t.get("start"),
                    "end": t.get("end"),
                    "metric": t.get("metric"),
                })
                m = self._jsonable_matrix(t.get("matrix"))
                if m is not None:
                    mats[str(tid)] = m
            payload["window_size"] = {
                "lag": ws.get("lag"),
                "curve": ws.get("curve") or {},
                "ticks": ticks,
                "extremes": ws.get("extremes") or {},
                "matrices": mats,
            }

        lg = sc.get("lag") or {}
        if isinstance(lg, dict) and (lg.get("curve") or lg.get("ticks")):
            mats = {}
            ticks = []
            for t in (lg.get("ticks") or []):
                tid = t.get("id")
                if not tid:
                    continue
                ticks.append({"id": tid, "lag": t.get("lag"), "metric": t.get("metric")})
                m = self._jsonable_matrix(t.get("matrix"))
                if m is not None:
                    mats[str(tid)] = m
            payload["lag"] = {
                "grid": lg.get("grid") or [],
                "curve": lg.get("curve") or {},
                "ticks": ticks,
                "extremes": lg.get("extremes") or {},
                "matrices": mats,
            }

        cube = sc.get("cube") or {}
        if isinstance(cube, dict) and (cube.get("points") or cube.get("gallery")):
            pts = []
            mats = {}
            for p in (cube.get("points") or []):
                pid = p.get("id")
                if not pid:
                    continue
                pts.append({
                    "id": pid,
                    "window_size": p.get("window_size"),
                    "lag": p.get("lag"),
                    "start": p.get("start"),
                    "end": p.get("end"),
                    "metric": p.get("metric"),
                    "tag": p.get("tag"),
                    "has_matrix": bool(p.get("matrix") is not None),
                })
                m = self._jsonable_matrix(p.get("matrix"))
                if m is not None:
                    mats[str(pid)] = m
            gallery = []
            for g in (cube.get("gallery") or []):
                gid = g.get("id")
                if not gid:
                    continue
                gallery.append({
                    "id": gid,
                    "window_size": g.get("window_size"),
                    "lag": g.get("lag"),
                    "start": g.get("start"),
                    "end": g.get("end"),
                    "metric": g.get("metric"),
                    "tag": g.get("tag"),
                })
                m = self._jsonable_matrix(g.get("matrix"))
                if m is not None:
                    mats[str(gid)] = m
            payload["cube"] = {
                "matrix_mode": cube.get("matrix_mode"),
                "matrix_limit": cube.get("matrix_limit"),
                "eval_limit": cube.get("eval_limit"),
                "combos": cube.get("combos") or [],
                "window_sizes": cube.get("window_sizes") or [],
                "lag_grid": cube.get("lag_grid") or [],
                "points": pts,
                "gallery": gallery,
                "selectable_ids": cube.get("selectable_ids") or [],
                "extremes": cube.get("extremes") or {},
                "matrices": mats,
            }
        return payload

    def _build_scans_interactive_html(self, payload: dict) -> str:
        """HTML блок со сканами + JSON payload для JS (Plotly)."""
        anchor = payload.get("anchor")
        if not anchor:
            return ""

        blocks: list[str] = []
        prefix = str(anchor)

        def _scan_block(title: str, key: str) -> str:
            return (
                f"<div class='scanblock'>"
                f"<h3>{html.escape(title)}</h3>"
                f"<div class='grid2'>"
                f"  <div><div class='scanplot' id='{prefix}_{key}_plot'></div></div>"
                f"  <div>"
                f"    <div class='scancontrols'>"
                f"      <button class='btn' id='{prefix}_{key}_best'>best</button>"
                f"      <button class='btn' id='{prefix}_{key}_median'>median</button>"
                f"      <button class='btn' id='{prefix}_{key}_worst'>worst</button>"
                f"      <select class='sel' id='{prefix}_{key}_sel'></select>"
                f"    </div>"
                f"    <div class='muted' id='{prefix}_{key}_meta'></div>"
                f"    <div class='scanheat' id='{prefix}_{key}_heat'></div>"
                f"  </div>"
                f"</div>"
                f"</div>"
            )

        if payload.get("window_pos"):
            blocks.append(_scan_block("Window position scan", "pos"))
        if payload.get("window_size"):
            blocks.append(_scan_block("Window size scan", "wsize"))
        if payload.get("lag"):
            blocks.append(_scan_block("Lag scan", "lag"))
        if payload.get("cube"):
            blocks.append(
                f"<div class='scanblock'>"
                f"<h3>Cube scan (window_size × lag × position)</h3>"
                f"<div class='grid2'>"
                f"  <div><div class='scanplot' id='{prefix}_cube_plot'></div></div>"
                f"  <div>"
                f"    <div class='scancontrols'>"
                f"      <button class='btn' id='{prefix}_cube_best'>best</button>"
                f"      <button class='btn' id='{prefix}_cube_median'>median</button>"
                f"      <button class='btn' id='{prefix}_cube_worst'>worst</button>"
                f"      <select class='sel' id='{prefix}_cube_sel'></select>"
                f"    </div>"
                f"    <div class='muted' id='{prefix}_cube_meta'></div>"
                f"    <div class='scanheat' id='{prefix}_cube_heat'></div>"
                f"  </div>"
                f"</div>"
                f"</div>"
            )
        if not blocks:
            return ""
        js = json.dumps(payload, ensure_ascii=False).replace("</", "<\/")
        return "".join(blocks) + f"<script type='application/json' id='scan_data_{prefix}'>{js}</script>"

    def _matrix_table(self, mat: np.ndarray, cols: list) -> str:
        if mat is None or not isinstance(mat, np.ndarray) or mat.size == 0:
            return "<div class='muted'>Нет данных</div>"
        rows = []
        header = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
        rows.append(f"<table class='matrix'><thead><tr><th></th>{header}</tr></thead><tbody>")

        for i, rname in enumerate(cols):
            cells = "".join(f"<td>{mat[i, j]:.4g}</td>" for j in range(len(cols)))
            rows.append(f"<tr><th>{html.escape(str(rname))}</th>{cells}</tr>")
        rows.append("</tbody></table>")
        return "".join(rows)

    def _carousel(self, items: List[Tuple[str, str]], cid: str) -> str:
        if not items:
            return "<div class='muted'>Нет диагностических графиков</div>"

        tabs = "".join(
            f"<button class='tab' onclick=\"showSlide('{cid}', {i})\">{html.escape(lbl)}</button>"
            for i, (lbl, _) in enumerate(items)
        )

        slides = "".join(
            f"<div class='slide' id='{cid}_s{i}' style='display:{'block' if i == 0 else 'none'}'>"
            f"<div class='slideLabel'>{html.escape(lbl)}</div>"
            f"<img src='data:image/png;base64,{b64}' />"
            f"</div>"
            for i, (lbl, b64) in enumerate(items)
        )
        return f"<div class='carousel'><div class='tabs'>{tabs}</div>{slides}</div>"

    def generate(self, output_path: str, **kwargs) -> str:
        """Основной метод построения отчета."""
        include_diagnostics = bool(kwargs.get("include_diagnostics", True))
        include_matrix_tables = kwargs.get("include_matrix_tables", True)
        include_fft_plots = bool(kwargs.get("include_fft_plots", False))
        include_scans = bool(kwargs.get("include_scans", True))
        include_series_files = bool(kwargs.get("include_series_files", True))
        series_preview_rows = int(kwargs.get("series_preview_rows", 200))
        harmonic_top_k = int(kwargs.get("harmonic_top_k", 5))
        graph_threshold = kwargs.get("graph_threshold", 0.2)
        p_alpha = kwargs.get("p_alpha", 0.05)

        df = self.tool.data_normalized if not self.tool.data_normalized.empty else self.tool.data
        # Предрасчёт компактных pairwise-таблиц для UI/секции отчёта.
        try:
            self.tool.build_pairwise_summaries(p_alpha=float(p_alpha))
        except Exception:
            pass
        cols = list(df.columns)
        variants = list(self.tool.results.keys())

        sections = []
        toc = []

        # Главный экран: raw/proc в едином масштабе + отчёт предобработки + гармоники.
        try:
            raw_df = getattr(self.tool, "data_raw", None)
            proc_df = getattr(self.tool, "data_preprocessed", None)
            if raw_df is None or getattr(raw_df, "empty", True):
                raw_df = self.tool.data
            if proc_df is None or getattr(proc_df, "empty", True):
                proc_df = self.tool.data

            y_domain = None
            try:
                vals = raw_df.to_numpy(dtype=float)
                if np.isfinite(vals).any():
                    y_domain = (float(np.nanmin(vals)), float(np.nanmax(vals)))
            except Exception:
                y_domain = None

            # Файлы рядов рядом с отчётом (удобно открыть/переиспользовать).
            series_files_html = ""
            try:
                if include_series_files:
                    out_dir = Path(output_path).parent
                    base = Path(output_path).stem
                    series_xlsx = out_dir / f"{base}_series.xlsx"
                    series_csv = out_dir / f"{base}_series.csv"

                    # Единый файл (xlsx) со слоями: RAW/PREPROCESSED/AFTER_AUTODIFF/(NORMALIZED)
                    try:
                        self.tool.export_series_bundle(str(series_xlsx))
                    except Exception:
                        raw_df.to_excel(series_xlsx, index=False)

                    # Плоский CSV (RAW)
                    try:
                        raw_df.to_csv(series_csv, index=False)
                    except Exception:
                        pass

                    links = []
                    try:
                        if series_xlsx.exists():
                            links.append(f"<a href='{html.escape(series_xlsx.name)}' download>{html.escape(series_xlsx.name)}</a>")
                        if series_csv.exists():
                            links.append(f"<a href='{html.escape(series_csv.name)}' download>{html.escape(series_csv.name)}</a>")
                    except Exception:
                        links = []

                    if links:
                        series_files_html = "<div class='meta'><b>Ряды (файлы):</b> " + " • ".join(links) + "</div>"
            except Exception:
                series_files_html = ""

            b64_raw = self._b64_png(plots.plot_timeseries_panel(raw_df, "Ряды: RAW (общий масштаб Y)", y_domain=y_domain))
            b64_proc = self._b64_png(plots.plot_timeseries_panel(proc_df, "Ряды: после предобработки/auto-diff (общий масштаб Y)", y_domain=y_domain))

            prep = {}
            try:
                prep = self.tool.get_preprocessing_summary()
            except Exception:
                prep = {}

            prep_lines = []
            p0 = prep.get("preprocess") or {}
            if p0.get("enabled") is not None:
                prep_lines.append(f"<b>Предобработка</b>: {'ON' if p0.get('enabled') else 'OFF'}")
            if p0.get("dropped_columns"):
                prep_lines.append("<b>Удалено</b>: " + html.escape(", ".join(map(str, p0.get("dropped_columns", [])))))
            if p0.get("steps_global"):
                prep_lines.append("<b>Шаги</b>: " + html.escape(" | ".join(map(str, p0.get("steps_global", [])[:12]))))
            ad = prep.get("autodiff") or {}
            if ad.get("enabled"):
                prep_lines.append("<b>Auto-diff</b>: дифференцированы " + html.escape(", ".join(map(str, ad.get("differenced", []))) or "—"))

            prep_html = "<div class='meta'>" + "<br/>".join(prep_lines) + "</div>" if prep_lines else ""

            harm = {}
            try:
                harm = self.tool.get_harmonics(top_k=harmonic_top_k)
            except Exception:
                harm = {}

            harm_cards = []
            for name, hk in (harm or {}).items():
                freqs = hk.get("freqs", [])
                amps = hk.get("amps", [])
                periods = hk.get("periods", [])
                lines = []
                for f, a, t in zip(freqs, amps, periods):
                    try:
                        lines.append(f"f={float(f):.5g}, A={float(a):.5g}, T={float(t):.5g}")
                    except Exception:
                        continue
                harm_cards.append(
                    "<div class='card'>"
                    f"<h3>{html.escape(str(name))}</h3>"
                    "<div class='muted'>Топ гармоники (FFT пики):</div>"
                    f"<div class='mono'>{html.escape(' | '.join(lines) if lines else '—')}</div>"
                    "</div>"
                )

            toc.insert(0, "<li><a href='#main'>Главный экран</a></li>")

            # Ряды по умолчанию скрыты: раскрываются по клику.
            preview_n = 0
            try:
                preview_n = min(int(series_preview_rows), int(raw_df.shape[0]))
            except Exception:
                preview_n = 0

            raw_preview_html = ""
            try:
                if preview_n > 0:
                    raw_preview_html = raw_df.head(preview_n).to_html(index=False, border=0)
            except Exception:
                raw_preview_html = ""

            series_details = (
                "<details><summary><b>Показать ряды (графики)</b></summary>"
                "<div class='grid2'>"
                f"<div><img src='data:image/png;base64,{b64_raw}'/></div>"
                f"<div><img src='data:image/png;base64,{b64_proc}'/></div>"
                "</div>"
                "</details>"
            )

            table_details = ""
            if raw_preview_html:
                table_details = (
                    "<details><summary><b>Показать таблицу рядов (preview)</b></summary>"
                    f"<div class='meta'>Показаны первые {preview_n} строк. Полные ряды — в файлах выше.</div>"
                    f"<div class='scroll'>{raw_preview_html}</div>"
                    "</details>"
                )

            sections.insert(
                0,
                "<section class='card' id='main'>"
                "<h1>Главный экран</h1>"
                "<div class='muted'>RAW/после предобработки в одном масштабе Y • применённая предобработка • гармоники</div>"
                f"{series_files_html}"
                f"{prep_html}"
                f"{series_details}"
                f"{table_details}"
                + "".join(harm_cards[:8])
                + "</section>"
            )
        except Exception:
            pass

        if include_diagnostics:
            toc.append("<li><a href='#diagnostics'>Первичный анализ</a></li>")
            diag = {}
            try:
                diag = self.tool.get_diagnostics()
            except Exception:
                diag = {}

            cards = []
            if not diag:
                cards.append("<div class='muted'>Нет диагностических данных</div>")
            else:
                for name, d in diag.items():
                    season = d.get("seasonality") or {}
                    fftp = d.get("fft_peaks") or {}

                    def _fmt(x) -> str:
                        if x is None:
                            return "—"
                        try:
                            if isinstance(x, (int, np.integer)):
                                return str(int(x))
                            if not np.isfinite(float(x)):
                                return "NaN"
                            return f"{float(x):.4g}"
                        except Exception:
                            return html.escape(str(x))

                    pk_freqs = fftp.get("freqs", [])
                    pk_periods = fftp.get("periods", [])
                    if pk_freqs:
                        pk_line = ", ".join(
                            f"f={_fmt(f)} (T={_fmt(t)})" for f, t in zip(pk_freqs, pk_periods)
                        )
                    else:
                        pk_line = "—"

                    fft_html = ""
                    if include_fft_plots:
                        try:
                            if getattr(self.tool, "data", None) is not None and name in getattr(self.tool, "data").columns:
                                fft_b64 = self._plot_fft_b64(self.tool.data[name], f"FFT spectrum: {name}", fs=float(getattr(self.tool, "fs", 1.0)))
                                fft_html = (
                                    "<div style='margin-top:10px'>"
                                    "<div class='muted'>FFT спектр</div>"
                                    f"<img style='max-width:100%;border-radius:10px' src='data:image/png;base64,{fft_b64}'/>"
                                    "</div>"
                                )
                        except Exception:
                            fft_html = ""

                    cards.append(
                        "<div class='card'>"
                        f"<h3>{html.escape(str(name))}</h3>"
                        "<div class='grid'>"
                        "<div><b>ADF p</b>: " + _fmt(d.get("adf_p")) + "</div>"
                        "<div><b>Hurst (R/S)</b>: " + _fmt(d.get("hurst_rs")) + "</div>"
                        "<div><b>Hurst (DFA)</b>: " + _fmt(d.get("hurst_dfa")) + "</div>"
                        "<div><b>Hurst (AggVar)</b>: " + _fmt(d.get("hurst_aggvar")) + "</div>"
                        "<div><b>Hurst (PSD)</b>: " + _fmt(d.get("hurst_wavelet")) + "</div>"
                        "<div><b>Sample entropy</b>: " + _fmt(d.get("sample_entropy")) + "</div>"
                        "<div><b>Shannon H</b>: " + _fmt(d.get("shannon_entropy")) + "</div>"
                        "<div><b>Permutation H</b>: " + _fmt(d.get("permutation_entropy")) + "</div>"
                        "<div><b>ACF сезонность</b>: период="
                        + _fmt(season.get("acf_period"))
                        + ", сила="
                        + _fmt(season.get("acf_strength"))
                        + "</div>"
                        + "<div><b>FFT пики</b>: " + html.escape(pk_line) + "</div>"
                        + fft_html
                        + "</div>"
                        + "</div>"
                    )

            sections.append(
                "<section class='card' id='diagnostics'>"
                "<h1>Первичный анализ рядов</h1>"
                "<div class='muted'>Стационарность • разные Hurst • сезонность • FFT пики • базовые энтропии</div>"
                + "".join(cards)
                + "</section>"
            )

        for k, variant in enumerate(variants, start=1):
            info = METHOD_INFO.get(variant, {"title": variant, "meaning": ""})
            anchor = f"m_{k}"
            toc.append(f"<li><a href='#{anchor}'>{html.escape(info['title'])}</a></li>")

            mat = self.tool.results.get(variant)
            chosen_lag = getattr(self.tool, "variant_lags", {}).get(variant, 1)

            meta = getattr(self.tool, "results_meta", {}).get(variant, {}) or {}
            meta_lines = []
            if meta.get("partial"):
                p = meta["partial"]
                if p.get("pairwise_policy") == "others":
                    meta_lines.append("Partial: для пары (Xi,Xj) исключено линейное влияние всех остальных переменных.")
                elif p.get("pairwise_policy") == "custom":
                    cc = p.get("custom_controls") or []
                    meta_lines.append("Partial: исключено влияние control=" + html.escape(", ".join(map(str, cc))) + ".")
                else:
                    meta_lines.append("Partial: контроль отключён.")

            if meta.get("lag_optimization"):
                lo = meta["lag_optimization"]
                meta_lines.append(
                    f"Lag: выбран автоматически (1..{int(lo.get('max_lag', 1))}), критерий: {html.escape(str(lo.get('criterion', '')))}."
                )

            win = meta.get("window")
            if win and win.get("best"):
                b = win["best"]
                meta_lines.append(
                    f"Окна: sizes={html.escape(str(win.get('sizes')))}; policy={html.escape(str(win.get('policy')))}; best window_size={int(b.get('window_size'))}, stride={int(b.get('stride'))}."
                )

            meta_html = ""
            if meta_lines:
                meta_html = "<div class='meta'>" + "<br/>".join(meta_lines) + "</div>"

            cube_html = ""
            cube = meta.get("window_cube") or {}
            if isinstance(cube, dict) and (cube.get("points") or []):
                try:
                    b64 = self._plot_cube3d_b64(cube.get("points") or [], f"{variant}: window×lag×position")
                    cube_html = "<div class='card'><h3>3D: window×lag×position</h3><img src='data:image/png;base64," + b64 + "'/></div>"
                except Exception:
                    cube_html = ""

            pair_table_html = ""
            try:
                dfp = (getattr(self.tool, "pairwise_summaries", {}) or {}).get(variant)
                if dfp is not None and not getattr(dfp, "empty", True):
                    if is_pvalue_method(variant):
                        view = dfp.sort_values("value", ascending=True).head(20)
                    else:
                        view = dfp.assign(absval=dfp["value"].abs()).sort_values("absval", ascending=False).head(20).drop(columns=["absval"])
                    rows = ["<table class='pairs'><thead><tr><th>src</th><th>tgt</th><th>value</th><th>flag</th></tr></thead><tbody>"]
                    for _, r in view.iterrows():
                        rows.append(f"<tr><td>{html.escape(str(r['src']))}</td><td>{html.escape(str(r['tgt']))}</td><td>{float(r['value']):.4g}</td><td>{html.escape(str(r.get('flag','')))}</td></tr>")
                    rows.append("</tbody></table>")
                    pair_table_html = "<div class='card'><h3>Сводка по парам (топ)</h3>" + "".join(rows) + "</div>"
            except Exception:
                pair_table_html = ""

            legend = f"Lag={chosen_lag}"
            buf_heat = plots.plot_heatmap(mat, f"{variant} Теплокарта", labels=cols, legend_text=legend)

            is_pval = is_pvalue_method(variant)
            is_dir = is_directed_method(variant)
            thr = p_alpha if is_pval else graph_threshold

            buf_conn = plots.plot_connectome(
                mat,
                f"{variant} Граф",
                threshold=thr,
                directed=is_dir,
                invert_threshold=is_pval,
                legend_text=f"{legend}, порог={thr}",
            )

            car_items = [
                ("Теплокарта", self._b64_png(buf_heat)),
                ("Граф связности", self._b64_png(buf_conn)),
            ]

            table_html = ""
            if include_matrix_tables:
                table_html = f"<h4>Матрица значений (Lag={chosen_lag})</h4>" + self._matrix_table(mat, cols)

            win_curve_html = ""
            try:
                wmeta = getattr(self.tool, "window_analysis", {}).get(variant)
                if wmeta and wmeta.get("best") and wmeta["best"].get("curve"):
                    curve = wmeta["best"]["curve"]
                    xs = curve.get("x", [])
                    ys = curve.get("y", [])
                    if xs and ys:
                        b64 = self._plot_curve_b64(xs, ys, f"{variant}: quality по окнам", "start")
                        win_curve_html = (
                            "<div style='margin-top:10px'>"
                            "<div class='muted'>Кривая качества по сдвигу окна (в точках)</div>"
                            f"<img style='max-width:100%;border-radius:10px' src='data:image/png;base64,{b64}'/>"
                            "</div>"
                        )
            except Exception:
                win_curve_html = ""

            scans_html = ""
            if include_scans:
                try:
                    sm = (getattr(self.tool, "results_meta", {}) or {}).get(variant, {}) or {}
                    sc = sm.get("window_scans") or {}
                    if isinstance(sc, dict):
                        payload = self._build_scan_payload(variant=variant, anchor=anchor, cols=cols, sc=sc)
                        scans_html = self._build_scans_interactive_html(payload)
                except Exception:
                    scans_html = ""

            sections.append(
                f"<section class='card' id='{anchor}'>"
                f"<h2>{html.escape(info['title'])}</h2>"
                f"<div class='muted'>{html.escape(info.get('meaning', ''))}</div>"
                f"{meta_html}{cube_html}{pair_table_html}"
                f"{self._carousel(car_items, f'c_{k}')}"
                f"{win_curve_html}"
                f"{scans_html}"
                f"{table_html}"
                f"</section>"
            )

        scan_js = """<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script>
function showSlide(cid, idx){
  const slides = document.querySelectorAll('[id^="'+cid+'_s"]');
  slides.forEach((el,i)=>{ el.style.display = (i===idx?'block':'none'); });
}
function _fmt(v){ return (v===null || v===undefined || Number.isNaN(Number(v))) ? '—' : Number(v).toFixed(4); }
function _renderHeat(id, labels, z){
  if(!window.Plotly){ return; }
  const d = document.getElementById(id); if(!d){ return; }
  Plotly.newPlot(d, [{type:'heatmap', z:z||[], x:labels||[], y:labels||[], colorscale:'RdBu'}], {margin:{t:10,r:10,b:40,l:40}});
}
function _renderLine(id, xs, ys, xTitle){
  if(!window.Plotly){ return; }
  const d = document.getElementById(id); if(!d){ return; }
  Plotly.newPlot(d, [{type:'scatter', mode:'lines+markers', x:xs||[], y:ys||[]}], {margin:{t:10,r:10,b:40,l:45}, xaxis:{title:xTitle}, yaxis:{title:'metric'}});
}
function _init1D(prefix, key, data, labels){
  const plotId = `${prefix}_${key}_plot`, heatId = `${prefix}_${key}_heat`, selId = `${prefix}_${key}_sel`, metaId = `${prefix}_${key}_meta`;
  _renderLine(plotId, (data.curve||{}).x||[], (data.curve||{}).y||[], key);
  const ticks = data.ticks||[]; const mats = data.matrices||{}; const ext = data.extremes||{};
  const sel = document.getElementById(selId); if(!sel){ return; }
  sel.innerHTML = ticks.map((t,i)=>`<option value="${t.id}">#${i} ${t.id}</option>`).join('');
  const show = (id)=>{ const t=(ticks.find(x=>x.id===id)||{}); _renderHeat(heatId, labels, mats[id]); const m=document.getElementById(metaId); if(m){ m.textContent = JSON.stringify(t); } if(sel.value!==id){ sel.value=id; } };
  sel.onchange = ()=>show(sel.value);
  ['best','median','worst'].forEach(tag=>{ const b=document.getElementById(`${prefix}_${key}_${tag}`); if(b){ b.onclick=()=>{ if(ext[tag]) show(ext[tag]); }; }});
  const first = ext.best || (ticks[0]||{}).id; if(first){ show(first); }
}
function _initCube(prefix, data, labels){
  if(!window.Plotly){ return; }
  const pts = data.points||[], mats=data.matrices||{}, ext=data.extremes||{};
  const plot = document.getElementById(`${prefix}_cube_plot`), sel=document.getElementById(`${prefix}_cube_sel`), meta=document.getElementById(`${prefix}_cube_meta`);
  if(!plot || !sel){ return; }
  Plotly.newPlot(plot, [{type:'scatter3d', mode:'markers', x:pts.map(p=>p.window_size), y:pts.map(p=>p.lag), z:pts.map(p=>p.start), text:pts.map(p=>`${p.id} q=${_fmt(p.metric)} ${p.tag||''}`), marker:{size:4,color:pts.map(p=>p.metric),colorscale:'Viridis'}}], {margin:{t:10,r:10,b:10,l:10}, scene:{xaxis:{title:'window'},yaxis:{title:'lag'},zaxis:{title:'start'}}});
  sel.innerHTML = pts.map((p,i)=>`<option value="${p.id}">#${i} ${p.id}</option>`).join('');
  const show=(id)=>{ _renderHeat(`${prefix}_cube_heat`, labels, mats[id]); const p=pts.find(x=>x.id===id)||{}; if(meta){ meta.textContent = JSON.stringify(p); } if(sel.value!==id){ sel.value=id; } };
  sel.onchange = ()=>show(sel.value);
  plot.on('plotly_click', (ev)=>{ const p=(ev.points||[])[0]; if(p){ const id=pts[p.pointIndex] && pts[p.pointIndex].id; if(id){ show(id); } }});
  ['best','median','worst'].forEach(tag=>{ const b=document.getElementById(`${prefix}_cube_${tag}`); if(b){ b.onclick=()=>{ if(ext[tag]) show(ext[tag]); }; }});
  const first = ext.best || (pts[0]||{}).id; if(first){ show(first); }
}
function _initScans(){
  document.querySelectorAll('script[id^="scan_data_"]').forEach((el)=>{
    try{
      const payload = JSON.parse(el.textContent||'{}'); const prefix=payload.anchor; const labels=payload.labels||[];
      if(payload.window_pos){ _init1D(prefix, 'pos', payload.window_pos, labels); }
      if(payload.window_size){ _init1D(prefix, 'wsize', payload.window_size, labels); }
      if(payload.lag){ _init1D(prefix, 'lag', payload.lag, labels); }
      if(payload.cube){ _initCube(prefix, payload.cube, labels); }
    }catch(e){ console.warn('scan init failed', e); }
  });
}
document.addEventListener('DOMContentLoaded', _initScans);
</script>"""

        html_content = f"""<!doctype html>
<html lang="ru">
<head>
<meta charset='utf-8'/>
<title>Отчет: Анализ временных рядов</title>
<style>
body{{font-family:Arial, sans-serif; margin:0; background:#fafafa;}}
header{{padding:16px 20px; background:#111; color:#fff;}}
main{{display:flex; gap:16px; padding:16px 20px;}}
nav{{width:260px; position:sticky; top:16px; align-self:flex-start; background:#fff; border:1px solid #ddd; border-radius:10px; padding:12px;}}
.card{{background:#fff; border:1px solid #ddd; border-radius:12px; padding:14px; margin-bottom:14px;}}
.muted{{color:#666; font-size:13px;}}
.carousel{{border:1px solid #eee; border-radius:12px; padding:10px; margin-top:10px;}}
.tabs{{display:flex; flex-wrap:wrap; gap:6px; margin-bottom:10px;}}
.tab{{border:1px solid #ccc; background:#f6f6f6; border-radius:15px; padding:6px 12px; cursor:pointer; font-size:12px;}}
.slide img{{max-width:100%; border-radius:10px;}}
img.inline{{max-width:100%;height:auto;border-radius:10px;}}
table.pairs{{width:100%;border-collapse:collapse;font-size:12px;}}
table.pairs th, table.pairs td{{border:1px solid #ddd;padding:6px;}}
table.pairs th{{background:#f5f5f5;text-align:left;}}
table.matrix{{border-collapse:collapse; font-size:11px; width:100%; overflow-x:auto; display:block;}}
table.matrix th, table.matrix td{{border:1px solid #eee; padding:4px 6px; text-align:right;}}
table.matrix th{{background:#f9f9f9; text-align:center;}}
.grid{{display:grid; grid-template-columns:repeat(auto-fit, minmax(220px,1fr)); gap:8px; margin-top:10px;}}
.meta{{margin-top:8px; padding:8px 10px; border:1px dashed #ddd; border-radius:10px; font-size:12px; color:#444; background:#fcfcfc;}}
.grid2{{display:grid; grid-template-columns:1fr 1fr; gap:12px;}}
.scroll{{max-height:420px; overflow:auto; border:1px solid #ddd; border-radius:10px; padding:8px; background:#fff;}}
.scroll table{{width:100%; border-collapse:collapse; font-size:12px;}}
.scroll th, .scroll td{{border-bottom:1px solid #eee; padding:4px 6px; text-align:left;}}
.mono{{font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-size:12px;}}
</style>
{scan_js}
</head>
<body>
<header>
  <div style='font-size:18px;font-weight:700;'>Отчет о связности временных рядов</div>
  <div class='muted' style='color:#ddd;'>Методов: {len(variants)} • Переменных: {len(cols)} • Точек: {len(df)}</div>
</header>
<main>
  <nav>
    <div style='font-weight:700;margin-bottom:8px;'>Оглавление</div>
    <ul>{''.join(toc)}</ul>
  </nav>
  <div style='flex:1; min-width:0;'>
    {''.join(sections)}
  </div>
</main>
</body>
</html>"""

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html_content, encoding="utf-8")
        return str(out)
