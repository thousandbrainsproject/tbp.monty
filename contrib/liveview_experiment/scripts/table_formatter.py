"""Table formatting utilities for complexity reports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from pathlib import Path

    from .analyze_complexity import FunctionMetrics


class TableFormatter:
    """Formats priority tables for complexity reports."""

    @staticmethod
    def calculate_column_widths(
        metrics_with_paths: list[tuple[Path, FunctionMetrics]],
    ) -> dict[str, int]:
        """Calculate column widths for table.

        Args:
            metrics_with_paths: List of (path, metric) tuples

        Returns:
            Dictionary of column widths
        """
        max_file_len = min(max(len(str(p)) for p, _ in metrics_with_paths), 50)
        max_func_len = min(max(len(m.function_name) for _, m in metrics_with_paths), 30)

        return {
            "priority": 12,
            "file": max_file_len + 2,
            "function": max_func_len + 2,
            "lines": 12,
            "nest": 6,
            "complex": 8,
            "length": 8,
            "params": 8,
        }

    @staticmethod
    def build_header(col_widths: dict[str, int]) -> str:
        """Build table header string.

        Args:
            col_widths: Column width dictionary

        Returns:
            Header string
        """
        return (
            f"{'Priority':<{col_widths['priority']}} "
            f"{'File':<{col_widths['file']}} "
            f"{'Function':<{col_widths['function']}} "
            f"{'Lines':<{col_widths['lines']}} "
            f"{'Nest':<{col_widths['nest']}} "
            f"{'Complex':<{col_widths['complex']}} "
            f"{'Length':<{col_widths['length']}} "
            f"{'Params':<{col_widths['params']}}"
        )

    @staticmethod
    def format_parameter_string(metric: FunctionMetrics) -> str:
        """Format parameter count string.

        Args:
            metric: Function metrics

        Returns:
            Formatted parameter string
        """
        param_str = str(metric.parameter_count)
        if metric.has_varargs:
            param_str += "+*args"
        if metric.has_kwargs:
            param_str += "+**kwargs"
        return param_str

    @staticmethod
    def format_row(
        rel_path: Path,
        metric: FunctionMetrics,
        col_widths: dict[str, int],
        format_priority: Callable[[float], str],
    ) -> str:
        """Format a single table row.

        Args:
            rel_path: Relative file path
            metric: Function metrics
            col_widths: Column width dictionary
            format_priority: Function to format priority score

        Returns:
            Formatted row string
        """
        file_str = str(rel_path)[: col_widths["file"] - 2]
        func_str = metric.function_name[: col_widths["function"] - 2]
        param_str = TableFormatter.format_parameter_string(metric)

        return (
            f"{format_priority(metric.priority_score):<{col_widths['priority']}} "
            f"{file_str:<{col_widths['file']}} "
            f"{func_str:<{col_widths['function']}} "
            f"{metric.line_start}-{metric.line_end:<{col_widths['lines']}} "
            f"{metric.max_nesting_level:<{col_widths['nest']}} "
            f"{metric.cyclomatic_complexity:<{col_widths['complex']}} "
            f"{metric.function_length:<{col_widths['length']}} "
            f"{param_str:<{col_widths['params']}}"
        )
