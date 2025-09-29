# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

import nh3

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationError:
    """Represents a validation error with file context."""

    def __init__(
        self,
        message: str,
        file_path: str,
        field: str,
    ):
        self.message = message
        self.file_path = file_path
        self.field = field


class RecordValidator:
    """Validates and transforms records for the future work widget."""

    COMMA_SEPARATED_FIELDS = ["tags", "owner", "skills"]
    MAX_COMMA_SEPARATED_ITEMS = 10
    # add in required values once the future work docs are populated.
    REQUIRED_FIELDS: list[str] = []

    REGEX_PATTERNS = {
        "owner": [r"[a-zA-Z0-9][a-zA-Z0-9-]{0,38}"],
        "rfc": [
            r"https://github\.com/thousandbrainsproject/tbp\.monty/.*",
            r"required",
            r"optional",
            r"not-required",
            r"unknown",
        ],
    }


    def __init__(self, docs_snippets_dir: Path):
        self.exact_values: dict[str, list[str]] = {}
        self._load_validation_files(docs_snippets_dir)

    def validate(
        self, record: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, list[ValidationError]]:
        """Validate a record.

        Args:
            record: The record to validate

        Returns:
            Tuple of (record or None if there are errors, list of validation errors)
        """
        if record.get("path1") != "future-work" or "path2" not in record:
            return None, []

        record_copy = record.copy()
        file_path = record.get("path")

        errors = []
        self._validate_comma_separated_fields(record_copy, file_path, errors)
        self._validate_required_fields(record_copy, file_path, errors)
        self._validate_field_values(record_copy, file_path, errors)
        return (None, errors) if errors else (record_copy, errors)

    def _validate_comma_separated_fields(
        self,
        record: dict[str, Any],
        file_path: str,
        errors: list[ValidationError],
    ) -> None:
        """Process comma-separated fields by validating and transforming them.

        Args:
            record: The record being transformed (modified in place)
            file_path: Path to the source file for error reporting
            errors: List to append validation errors to
        """
        for field in self.COMMA_SEPARATED_FIELDS:
            if field in record and isinstance(record[field], str):
                items = [tag.strip() for tag in record[field].split(",")]
                if len(items) > self.MAX_COMMA_SEPARATED_ITEMS:
                    self._add_validation_error(
                        f"{field} field cannot have more than "
                        f"{self.MAX_COMMA_SEPARATED_ITEMS} items. "
                        f"Got {len(items)} items: {', '.join(items)}",
                        file_path,
                        field,
                        errors,
                    )
                record[field] = items

    def _validate_required_fields(
        self, record: dict[str, Any], file_path: str, errors: list[ValidationError]
    ) -> None:
        """Validate that all required fields are present and not empty."""
        for field in self.REQUIRED_FIELDS:
            if field not in record:
                self._add_validation_error(
                    f"Required field '{field}' is missing", file_path, field, errors
                )
                continue

            if not isinstance(record[field], str):
                self._add_validation_error(
                    f"Required field '{field}' must be a string, "
                    f"got {type(record[field]).__name__}",
                    file_path,
                    field,
                    errors,
                )
                continue

            if not record[field].strip():
                self._add_validation_error(
                    f"Required field '{field}' cannot be empty",
                    file_path,
                    field,
                    errors,
                )
                continue

    def _add_validation_error(
        self, message: str, file_path: str, field: str, errors: list[ValidationError]
    ) -> None:
        """Add a ValidationError to the errors list."""
        errors.append(ValidationError(message, file_path, field=field))

    def _extract_readable_values(self, field_name: str) -> list[str]:
        """Extract readable values for error messages.

        Args:
            field_name: Name of the field to get readable values for

        Returns:
            List of readable values for error messages
        """
        # For exact values, the values themselves are readable
        if field_name in self.exact_values:
            return sorted(self.exact_values[field_name])

        # For regex patterns, just return empty list
        return []

    def _load_validation_files(self, docs_snippets_dir: Path) -> None:
        """Load validation files from docs/snippets directory.

        Args:
            docs_snippets_dir: Path to the docs/snippets directory
        """
        future_work_files = list(docs_snippets_dir.glob("future-work-*.md"))

        for file_path in future_work_files:
            field_name = file_path.stem.replace("future-work-", "")

            if field_name in self.REGEX_PATTERNS:
                logger.debug(
                    f"Skipping {file_path.name} - using hardcoded regex "
                    f"patterns for '{field_name}'"
                )
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            simple_values = []
            for raw_item in content.split("`"):
                clean_item = nh3.clean(raw_item).strip()
                if clean_item:
                    simple_values.append(clean_item)

            if simple_values:
                self.exact_values[field_name] = simple_values
                logger.debug(
                    f"Loaded {len(simple_values)} exact match values for "
                    f"'{field_name}' from {file_path.name}"
                )

    def _get_all_validation_fields(self) -> set[str]:
        exact_keys = set(self.exact_values.keys())
        regex_keys = set(self.REGEX_PATTERNS.keys())
        return exact_keys | regex_keys

    def _normalize_values_to_list(self, record_values: Any) -> list[Any]:
        """Convert single values to list format for consistent processing.

        Args:
            record_values: The values from the record (single value or list)

        Returns:
            List of values to validate
        """
        return record_values if isinstance(record_values, list) else [record_values]

    def _is_value_valid(self, field_name: str, sanitized_value: str) -> bool:
        """Check if a value is valid for a given field.

        Args:
            field_name: The field being validated
            sanitized_value: The cleaned value to check

        Returns:
            True if the value is valid, False otherwise
        """
        if field_name in self.exact_values:
            if sanitized_value in self.exact_values[field_name]:
                return True

        if field_name in self.REGEX_PATTERNS:
            return any(
                re.fullmatch(pattern, sanitized_value)
                for pattern in self.REGEX_PATTERNS[field_name]
            )

        return False

    def _create_validation_error_message(
        self, field_name: str, sanitized_value: str
    ) -> str:
        """Create an appropriate error message for an invalid field value.

        Args:
            field_name: The field that failed validation
            sanitized_value: The invalid value

        Returns:
            Error message string
        """
        readable_values = self._extract_readable_values(field_name)
        if readable_values:
            values_str = ", ".join(readable_values)
            return (
                f"Invalid {field_name} value '{sanitized_value}'. "
                f"Valid values are: {values_str}"
            )
        return f"Invalid {field_name} value '{sanitized_value}'"

    def _validate_field_values(
        self, record: dict[str, Any], file_path: str, errors: list[ValidationError]
    ) -> None:
        """Validate field values against exact matches and regex patterns.

        Args:
            record: The record to validate
            file_path: Path to the source file for error reporting
            errors: List to append validation errors to
        """
        all_field_names = self._get_all_validation_fields()

        for field_name in all_field_names:
            if field_name not in record:
                continue

            values_to_check = (
                record[field_name]
                if isinstance(record[field_name], list)
                else [record[field_name]]
            )

            for value in values_to_check:
                sanitized_value = nh3.clean(str(value)).strip()

                if not self._is_value_valid(field_name, sanitized_value):
                    message = self._create_validation_error_message(
                        field_name, sanitized_value
                    )
                    self._add_validation_error(message, file_path, field_name, errors)
