"""HTML report generator abstraction for BigMasterTool."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class HTMLReportGenerator:
    """Generate HTML report using data and plots prepared by BigMasterTool."""

    tool: object

    def generate(self, output_path: str, **kwargs) -> str:
        """Build HTML report by delegating to the engine implementation hook."""
        return self.tool._export_html_report_impl(output_path, **kwargs)
