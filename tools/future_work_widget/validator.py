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
        """Validate and transform a record.

        Args:
            record: The record to validate and transform

        Returns:
            Tuple of (transformed record if valid or None if invalid,
            list of validation errors)
        """
        if record.get("path1") != "future-work" or "path2" not in record:
            return None, []

        transformed_record = record.copy()
        file_path = record.get("path")

        errors = []
        for field in self.COMMA_SEPARATED_FIELDS:
            if field in transformed_record and isinstance(
                transformed_record[field], str
            ):
                items = [tag.strip() for tag in transformed_record[field].split(",")]
                if len(items) > self.MAX_COMMA_SEPARATED_ITEMS:
                    self._add_validation_error(
                        f"{field} field cannot have more than "
                        f"{self.MAX_COMMA_SEPARATED_ITEMS} items. "
                        f"Got {len(items)} items: {', '.join(items)}",
                        file_path,
                        field,
                        errors,
                    )
                transformed_record[field] = items
        self._validate_required_fields(transformed_record, file_path, errors)
        self._validate_field_values(transformed_record, file_path, errors)

        return transformed_record, errors

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

    def _validate_field_values(
        self, record: dict[str, Any], file_path: str, errors: list[ValidationError]
    ) -> None:
        """Validate field values against exact matches and regex patterns.

        Args:
            record: The record to validate
            file_path: Path to the source file for error reporting
            errors: List to append validation errors to
        """
        # Check all fields that have validation rules
        exact_keys = set(self.exact_values.keys())
        regex_keys = set(self.REGEX_PATTERNS.keys())
        all_field_names = exact_keys | regex_keys

        for field_name in all_field_names:
            if field_name in record:
                record_values = record[field_name]

                values_to_check = (
                    record_values
                    if isinstance(record_values, list)
                    else [record_values]
                )

                for value in values_to_check:
                    sanitized_value = nh3.clean(str(value)).strip()
                    is_valid = False

                    if field_name in self.exact_values:
                        is_valid = sanitized_value in self.exact_values[field_name]

                    if not is_valid and field_name in self.REGEX_PATTERNS:
                        is_valid = any(
                            re.fullmatch(pattern, sanitized_value)
                            for pattern in self.REGEX_PATTERNS[field_name]
                        )

                    if not is_valid:
                        readable_values = self._extract_readable_values(field_name)
                        if readable_values:
                            values_str = ", ".join(readable_values)
                            message = (
                                f"Invalid {field_name} value '{sanitized_value}'. "
                                f"Valid values are: {values_str}"
                            )
                        else:
                            message = f"Invalid {field_name} value '{sanitized_value}'"

                        self._add_validation_error(
                            message, file_path, field_name, errors
                        )
