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
        cols = list(df.columns)
        variants = list(self.tool.results.keys())

        sections = []
        toc = []

        if include_diagnostics:
            toc.append("<li><a href='#diagnostics'>Первичный анализ</a></li>")
            cards = []
            cards.append("<div class='card'><h3>Диагностика данных</h3><p>Раздел сформирован.</p></div>")

            sections.append(
                "<section class='card' id='diagnostics'>"
                "<h1>Первичный анализ рядов</h1>"
                + "".join(cards)
                + "</section>"
            )

        for k, variant in enumerate(variants, start=1):
            info = METHOD_INFO.get(variant, {"title": variant, "meaning": ""})
            anchor = f"m_{k}"
            toc.append(f"<li><a href='#{anchor}'>{html.escape(info['title'])}</a></li>")

            mat = self.tool.results.get(variant)
            chosen_lag = self.tool.variant_lags.get(variant, 1)

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

            sections.append(
                f"<section class='card' id='{anchor}'>"
                f"<h2>{html.escape(info['title'])}</h2>"
                f"<div class='muted'>{html.escape(info.get('meaning', ''))}</div>"
                f"{self._carousel(car_items, f'c_{k}')}"
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
