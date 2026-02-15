"""
Генератор Excel-отчетов (локализованный).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from openpyxl.utils.dataframe import dataframe_to_rows

from src.config import is_directed_method, is_pvalue_method
from src.visualization import plots


@dataclass(slots=True)
class ExcelReportWriter:
    """Класс для записи результатов анализа в Excel файл."""

    tool: object

    def _add_image_to_sheet(self, ws, buf, cell: str, w: int = 400, h: int = 300) -> None:
        img = Image(buf)
        img.width = w
        img.height = h
        ws.add_image(img, cell)

    def write(self, save_path: str, **kwargs) -> str:
        """Создает и сохраняет Excel-файл со всеми результатами."""
        wb = Workbook()

        ws_data = wb.active
        ws_data.title = "Исходные данные"
        ws_data.append(list(self.tool.data.columns))
        for row in dataframe_to_rows(self.tool.data, index=False, header=False):
            ws_data.append(row)

        threshold = kwargs.get("threshold", 0.2)
        p_alpha = kwargs.get("p_value_alpha", 0.05)

        for variant, mat in self.tool.results.items():
            if mat is None:
                continue

            ws = wb.create_sheet(title=variant[:30])
            ws.append([f"Метод: {variant}"])
            ws.append(["Матрица значений:"])

            df_mat = pd.DataFrame(mat, columns=self.tool.data.columns, index=self.tool.data.columns)
            for row in dataframe_to_rows(df_mat, index=True, header=True):
                ws.append(row)

            last_row = ws.max_row + 2

            is_pval = is_pvalue_method(variant)
            thr = p_alpha if is_pval else threshold

            buf_heat = plots.plot_heatmap(mat, f"{variant} Теплокарта")
            self._add_image_to_sheet(ws, buf_heat, f"A{last_row}")

            buf_conn = plots.plot_connectome(
                mat,
                f"{variant} Граф",
                threshold=thr,
                directed=is_directed_method(variant),
                invert_threshold=is_pval,
            )
            self._add_image_to_sheet(ws, buf_conn, f"G{last_row}")

        wb.save(save_path)
        logging.info("[Excel] Отчет сохранен: %s", save_path)
        return save_path
