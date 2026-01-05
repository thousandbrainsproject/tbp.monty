"""Report generation utilities for complexity analysis."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from pathlib import Path

    from .analyze_complexity import FunctionMetrics


class ReportGenerator:
    """Generates complexity analysis reports."""

    @staticmethod
    def build_report_data(
        all_metrics: list[FunctionMetrics],
        regular_metrics: list[FunctionMetrics],
        protocol_metrics: list[FunctionMetrics],
        count_violations: Callable[[list[FunctionMetrics], str], int],
    ) -> dict[str, Any]:
        """Build report data dictionary.

        Args:
            all_metrics: All function metrics
            regular_metrics: Regular function metrics
            protocol_metrics: Protocol method metrics
            count_violations: Function to count violations

        Returns:
            Report data dictionary
        """
        return {
            "summary": {
                "total_functions": len(all_metrics),
                "regular_functions": len(regular_metrics),
                "protocol_methods": len(protocol_metrics),
                "functions_with_nesting_violations": count_violations(
                    regular_metrics, "nesting"
                ),
                "functions_with_high_complexity": count_violations(
                    regular_metrics, "complexity"
                ),
                "functions_with_high_length": count_violations(
                    regular_metrics, "length"
                ),
                "functions_with_too_many_parameters": count_violations(
                    regular_metrics, "parameters"
                ),
            },
            "functions": ReportGenerator._build_function_list(all_metrics),
        }

    @staticmethod
    def _build_function_list(
        all_metrics: list[FunctionMetrics],
    ) -> list[dict[str, Any]]:
        """Build list of function data for report.

        Args:
            all_metrics: All function metrics

        Returns:
            List of function dictionaries
        """
        return [
            {
                "file": m.file_path,
                "function": m.function_name,
                "line_start": m.line_start,
                "line_end": m.line_end,
                "cyclomatic_complexity": m.cyclomatic_complexity,
                "max_nesting_level": m.max_nesting_level,
                "function_length": m.function_length,
                "parameter_count": m.parameter_count,
                "priority_score": m.priority_score,
            }
            for m in all_metrics
            if m.priority_score > 0
            or (m.is_protocol_method and m.parameter_violation > 0)
        ]

    @staticmethod
    def save_report(report_path: Path, report_data: dict[str, Any]) -> None:
        """Save report to JSON file.

        Args:
            report_path: Path to save report
            report_data: Report data dictionary
        """
        with report_path.open("w") as f:
            json.dump(report_data, f, indent=2)
