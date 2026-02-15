"""Excel report writer abstraction for BigMasterTool."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ExcelReportWriter:
    """Generate Excel workbook report from tool state."""

    tool: object

    def write(self, save_path: str, **kwargs) -> str:
        """Write Excel report by delegating to the engine implementation hook."""
        return self.tool._export_big_excel_impl(save_path, **kwargs)
