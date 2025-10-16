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

    # fields that can have comma separated values in them.
    COMMA_SEPARATED_FIELDS = ["tags", "skills"]
    MAX_COMMA_SEPARATED_ITEMS = 10

    # these fields have custom logic to process them
    CUSTOM_VALIDATION_FIELDS = ["rfc", "contributor"]

    # add in required values once the future work docs are populated.
    REQUIRED_FIELDS: list[str] = []

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

        errors: list[ValidationError] = []

        if "path" not in record:
            errors.append(
                ValidationError(
                    "Record is missing required 'path' field", "unknown", "path"
                )
            )
            return None, errors

        file_path = str(record["path"])

        self._validate_comma_separated_fields(record_copy, file_path, errors)
        self._validate_required_fields(record_copy, file_path, errors)
        self._validate_custom_fields(record_copy, file_path, errors)
        self._validate_field_values(record_copy, file_path, errors)
        return (None if errors else record_copy, errors)

    def _process_comma_separated_field(
        self,
        field_name: str,
        field_value: str,
        file_path: str,
        errors: list[ValidationError],
    ) -> list[str] | None:
        """Process a comma-separated field and validate its length.

        Args:
            field_name: Name of the field being processed
            field_value: The comma-separated string value
            file_path: Path to the source file for error reporting
            errors: List to append validation errors to

        Returns:
            List of processed items, or None if validation failed
        """
        items = [item.strip() for item in field_value.split(",")]
        if len(items) > self.MAX_COMMA_SEPARATED_ITEMS:
            self._add_validation_error(
                f"{field_name} field cannot have more than "
                f"{self.MAX_COMMA_SEPARATED_ITEMS} items. "
                f"Got {len(items)} items: {', '.join(items)}",
                file_path,
                field_name,
                errors,
            )
            return None
        return items

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
            if field in record:
                field_value = str(record[field])

                processed_items = self._process_comma_separated_field(
                    field, field_value, file_path, errors
                )
                if processed_items is not None:
                    record[field] = processed_items

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

    def _validate_custom_fields(
        self, record: dict[str, Any], file_path: str, errors: list[ValidationError]
    ) -> None:
        """Validate fields that use custom validation logic."""
        for field_name in self.CUSTOM_VALIDATION_FIELDS:
            if field_name not in record:
                continue

            if field_name == "rfc":
                self._validate_rfc(record, file_path, errors)
            elif field_name == "contributor":
                self._validate_contributor(record, file_path, errors)

    def _validate_rfc(
        self, record: dict[str, Any], file_path: str, errors: list[ValidationError]
    ) -> None:
        """Validate the rfc field against its specific patterns."""
        if "rfc" not in record:
            return

        rfc_value = record["rfc"]
        if not isinstance(rfc_value, str):
            self._add_validation_error(
                f"rfc field must be a string, got {type(rfc_value).__name__}",
                file_path,
                "rfc",
                errors,
            )
            return

        sanitized_value = nh3.clean(rfc_value).strip()
        rfc_patterns = [
            r"https://github\.com/thousandbrainsproject/tbp\.monty/.*",
            r"required",
            r"optional",
            r"not-required",
        ]

        if not any(re.fullmatch(pattern, sanitized_value) for pattern in rfc_patterns):
            valid_options = "a GitHub URL, 'required', 'optional', or 'not-required'"
            self._add_validation_error(
                f"Invalid rfc value '{sanitized_value}'. Must be {valid_options}",
                file_path,
                "rfc",
                errors,
            )

    def _validate_contributor(
        self, record: dict[str, Any], file_path: str, errors: list[ValidationError]
    ) -> None:
        """Validate contributor field as comma-separated GitHub usernames."""
        if "contributor" not in record:
            return

        contributor_value = record["contributor"]
        if not isinstance(contributor_value, str):
            self._add_validation_error(
                f"contributor field must be a string",
                file_path,
                "contributor",
                errors,
            )
            return

        contributors = self._process_comma_separated_field(
            "contributor", contributor_value, file_path, errors
        )
        if contributors is None:
            return

        github_contributor_pattern = r"[a-zA-Z0-9][a-zA-Z0-9-]{0,38}"
        invalid_contributors = []

        for contributor in contributors:
            sanitized_contributor = nh3.clean(contributor).strip()
            if not re.fullmatch(github_contributor_pattern, sanitized_contributor):
                invalid_contributors.append(sanitized_contributor)

        if invalid_contributors:
            self._add_validation_error(
                f"Invalid contributor username(s): {', '.join(invalid_contributors)}. "
                f"Must be valid GitHub usernames (1-39 characters, "
                f"alphanumeric and hyphens, cannot start with hyphen)",
                file_path,
                "contributor",
                errors,
            )
            return

        record["contributor"] = contributors

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

        Raises:
            ValueError: If a validation file exists for a field that uses
                custom validation logic
        """
        future_work_files = list(docs_snippets_dir.glob("future-work-*.md"))

        for file_path in future_work_files:
            field_name = file_path.stem.replace("future-work-", "")

            if field_name in self.CUSTOM_VALIDATION_FIELDS:
                error_msg = (
                    f"Configuration error: {file_path.name} should not exist. "
                    f"The '{field_name}' field uses custom validation logic."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

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
        return exact_keys - set(self.CUSTOM_VALIDATION_FIELDS)

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
            return sanitized_value in self.exact_values[field_name]

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
