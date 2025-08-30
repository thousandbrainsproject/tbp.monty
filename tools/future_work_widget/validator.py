# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


class RecordValidator:
    """Validates and transforms records for the future work widget."""

    COMMA_SEPARATED_FIELDS = ["tags", "owner", "skills"]
    MAX_COMMA_SEPARATED_ITEMS = 10
    REQUIRED_FIELDS: List[str] = []  # add in rfc and estimated-scope once ready.

    def __init__(self, docs_snippets_dir: Optional[str] = None):
        self.errors: List[str] = []
        self.validation_sets: Dict[str, List[str]] = {}
        if docs_snippets_dir:
            self._load_validation_files(docs_snippets_dir)

    def validate(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate and transform a record.

        Args:
            record: The record to validate and transform

        Returns:
            The transformed record if valid, None if invalid
        """
        if record.get("path1") != "future-work" or "path2" not in record:
            return None

        transformed_record = record.copy()

        for field in self.COMMA_SEPARATED_FIELDS:
            if field in transformed_record and isinstance(
                transformed_record[field], str
            ):
                items = [tag.strip() for tag in transformed_record[field].split(",")]
                if len(items) > self.MAX_COMMA_SEPARATED_ITEMS:
                    self.errors.append(
                        f"{field} field cannot have more than "
                        f"{self.MAX_COMMA_SEPARATED_ITEMS} items. "
                        f"Got {len(items)} items: {', '.join(items)}"
                    )
                transformed_record[field] = items

        self._validate_required_fields(transformed_record)
        self._validate_field_values(transformed_record)

        return transformed_record

    def _validate_required_fields(self, record: Dict[str, Any]) -> None:
        """Validate that all required fields are present and not empty."""
        for field in self.REQUIRED_FIELDS:
            if field not in record:
                self.errors.append(f"Required field '{field}' is missing")
            elif not isinstance(record[field], str) or not record[field].strip():
                self.errors.append(f"Required field '{field}' cannot be empty")

    def _extract_readable_values(self, regex_patterns: List[str]) -> List[str]:
        """Extract human-readable values from regex patterns for error messages.

        Args:
            regex_patterns: List of regex patterns

        Returns:
            List of readable values for error messages
        """
        readable_values = []
        for pattern in regex_patterns:
            if pattern.startswith("\\b") and pattern.endswith("\\b"):
                escaped_word = pattern[2:-2]
                unescaped_word = escaped_word.replace("\\-", "-").replace("\\.", ".")
                readable_values.append(unescaped_word)
            else:
                readable_values.append(f"pattern: {pattern}")
        return sorted(readable_values)

    def get_errors(self) -> List[str]:
        """Get all validation errors.

        Returns:
            List of validation error messages
        """
        return self.errors.copy()

    def clear_errors(self) -> None:
        """Clear all validation errors."""
        self.errors.clear()

    def _load_validation_files(self, docs_snippets_dir: str) -> None:
        """Load validation files from docs/snippets directory.

        Args:
            docs_snippets_dir: Path to the docs/snippets directory
        """
        snippets_path = Path(docs_snippets_dir)
        if not snippets_path.exists():
            logging.warning(f"Snippets directory not found: {docs_snippets_dir}")
            return

        future_work_files = list(snippets_path.glob("future-work-*.md"))

        for file_path in future_work_files:
            field_name = file_path.stem.replace("future-work-", "")

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                regex_patterns = []
                for raw_item in content.split("`"):
                    clean_item = raw_item.strip()
                    if clean_item:
                        if re.match(r"^[a-zA-Z0-9-]+$", clean_item):
                            escaped_item = re.escape(clean_item).replace("\\-", "-")
                            regex_patterns.append(f"\\b{escaped_item}\\b")
                        else:
                            regex_patterns.append(clean_item)

                if regex_patterns:
                    self.validation_sets[field_name] = regex_patterns
                    logging.info(
                        f"Loaded {len(regex_patterns)} regex patterns for "
                        f"'{field_name}' from {file_path.name}"
                    )

            except (OSError, UnicodeDecodeError):
                logging.exception(f"Failed to load validation file {file_path}")

    def _validate_field_values(self, record: Dict[str, Any]) -> None:
        """Validate field values against loaded validation sets.

        Args:
            record: The record to validate
        """
        for field_name, regex_patterns in self.validation_sets.items():
            if field_name in record:
                record_values = record[field_name]

                values_to_check = (
                    record_values
                    if isinstance(record_values, list)
                    else [record_values]
                )

                for value in values_to_check:
                    if not any(
                        re.fullmatch(pattern, value) for pattern in regex_patterns
                    ):
                        readable_values = self._extract_readable_values(regex_patterns)
                        values_str = ", ".join(readable_values)

                        self.errors.append(
                            f"Invalid {field_name} value '{value}'. "
                            f"Valid values are: {values_str}"
                        )
