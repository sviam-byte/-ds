"""
Генератор HTML-отчетов для BigMasterTool.
"""

from __future__ import annotations

import base64
import html
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
        include_diagnostics = kwargs.get("include_diagnostics", True)
        include_matrix_tables = kwargs.get("include_matrix_tables", True)
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
                harm = self.tool.get_harmonics(top_k=5)
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
            sections.insert(
                0,
                "<section class='card' id='main'>"
                "<h1>Главный экран</h1>"
                "<div class='muted'>RAW/после предобработки в одном масштабе Y • применённая предобработка • гармоники</div>"
                f"{prep_html}"
                "<div class='grid2'>"
                f"<div><img src='data:image/png;base64,{b64_raw}'/></div>"
                f"<div><img src='data:image/png;base64,{b64_proc}'/></div>"
                "</div>"
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
                        "<div><b>FFT пики</b>: " + html.escape(pk_line) + "</div>"
                        "</div>"
                        "</div>"
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

            sections.append(
                f"<section class='card' id='{anchor}'>"
                f"<h2>{html.escape(info['title'])}</h2>"
                f"<div class='muted'>{html.escape(info.get('meaning', ''))}</div>"
                f"{meta_html}"
                f"{self._carousel(car_items, f'c_{k}')}"
                f"{win_curve_html}"
                f"{table_html}"
                f"</section>"
            )

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
table.matrix{{border-collapse:collapse; font-size:11px; width:100%; overflow-x:auto; display:block;}}
table.matrix th, table.matrix td{{border:1px solid #eee; padding:4px 6px; text-align:right;}}
table.matrix th{{background:#f9f9f9; text-align:center;}}
.grid{{display:grid; grid-template-columns:repeat(auto-fit, minmax(220px,1fr)); gap:8px; margin-top:10px;}}
.meta{{margin-top:8px; padding:8px 10px; border:1px dashed #ddd; border-radius:10px; font-size:12px; color:#444; background:#fcfcfc;}}
.grid2{{display:grid; grid-template-columns:1fr 1fr; gap:12px;}}
.mono{{font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-size:12px;}}
</style>
<script>
function showSlide(cid, idx){{
  const slides = document.querySelectorAll('[id^="'+cid+'_s"]');
  slides.forEach((el,i)=>{{ el.style.display = (i===idx?'block':'none'); }});
}}
</script>
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
