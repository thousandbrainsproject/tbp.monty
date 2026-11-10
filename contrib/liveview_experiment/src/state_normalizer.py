"""Normalizes experiment state values for template rendering."""

from __future__ import annotations

from .experiment_state import ExperimentState
from .state_field_builder import StateFieldBuilder


class StateNormalizer:
    """Normalizes experiment state values for template rendering."""

    @staticmethod
    def normalize(state: ExperimentState, normalized_status: str) -> ExperimentState:
        """Create normalized state copy for socket context.

        Args:
            state: Source experiment state
            normalized_status: Normalized status string

        Returns:
            New normalized ExperimentState instance
        """
        numeric = StateFieldBuilder.build_numeric_fields(state)
        string = StateFieldBuilder.build_string_fields(state)
        optional = StateFieldBuilder.build_optional_fields(state)
        boolean = StateFieldBuilder.build_boolean_fields(state)
        complex_fields = StateFieldBuilder.build_complex_fields(state)

        all_fields = {**numeric, **string, **optional, **boolean, **complex_fields}
        all_fields["status"] = normalized_status

        return ExperimentState(**all_fields)

    @staticmethod
    def _safe_int(value: int | None) -> int:
        """Convert value to int, defaulting to 0 if None.

        Args:
            value: Value to convert

        Returns:
            Integer value or 0
        """
        return value if value is not None else 0

    @staticmethod
    def _safe_str(value: str | None, default: str = "") -> str:
        """Convert value to string, defaulting to default if None.

        Args:
            value: Value to convert
            default: Default value if None

        Returns:
            String value or default
        """
        return value if value is not None else default

    @staticmethod
    def _safe_bool(value: bool | None) -> bool:
        """Convert value to bool, defaulting to False if None.

        Args:
            value: Value to convert

        Returns:
            Boolean value or False
        """
        return value if value is not None else False
