# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


class RecordValidator:
    """Validates and transforms records for the future work widget."""

    COMMA_SEPARATED_FIELDS = ["tags", "owner", "skills"]
    MAX_COMMA_SEPARATED_ITEMS = 10
    REQUIRED_FIELDS = ["estimated-scope", "rfc"]

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
            if (field in transformed_record
                and isinstance(transformed_record[field], str)):
                items = [
                    tag.strip() for tag in transformed_record[field].split(",")
                ]
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

    def _extract_readable_values(
        self, regex_patterns: List[str], field_name: str
    ) -> List[str]:
        """Extract human-readable values from regex patterns for error messages.

        Args:
            regex_patterns: List of regex patterns
            field_name: Name of the field being validated

        Returns:
            List of readable values for error messages
        """
        readable_values = []
        for pattern in regex_patterns:
            if pattern.startswith("\\b") and pattern.endswith("\\b"):
                # Extract simple word from \bword\b and unescape it
                escaped_word = pattern[2:-2]
                # Unescape common escaped characters
                unescaped_word = escaped_word.replace("\\-", "-").replace("\\.", ".")
                readable_values.append(unescaped_word)
            elif field_name == "rfc" and "github" in pattern:
                readable_values.append("valid RFC URL")
            else:
                # Show pattern description for complex patterns
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

                # All fields now support regex patterns
                regex_patterns = []
                for raw_item in content.split("`"):
                    clean_item = raw_item.strip()
                    if clean_item:
                        # If it looks like a simple word, wrap with word boundaries
                        if re.match(r"^[a-zA-Z0-9-]+$", clean_item):
                            # Don't escape hyphens in simple words
                            escaped_item = re.escape(clean_item).replace("\\-", "-")
                            regex_patterns.append(f"\\b{escaped_item}\\b")
                        else:
                            # Treat as regex pattern (for URLs, special patterns, etc.)
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

                if isinstance(record_values, list):
                    # Handle comma-separated fields (tags, skills, owner)
                    for value in record_values:
                        is_valid = any(
                            re.fullmatch(pattern, value) for pattern in regex_patterns
                        )
                        if not is_valid:
                            readable_values = self._extract_readable_values(
                                regex_patterns, field_name
                            )
                            values_str = ", ".join(readable_values)
                            self.errors.append(
                                f"Invalid {field_name} value '{value}'. "
                                f"Valid values are: {values_str}"
                            )
                elif isinstance(record_values, str):
                    # Handle single-value fields (rfc, status, estimated-scope)
                    is_valid = any(
                        re.fullmatch(pattern, record_values)
                        for pattern in regex_patterns
                    )
                    if not is_valid:
                        readable_values = self._extract_readable_values(
                            regex_patterns, field_name
                        )
                        values_str = ", ".join(readable_values)
                        if field_name == "rfc":
                            self.errors.append(
                                f"rfc must be one of: {values_str}. "
                                f"Got: {record_values}"
                            )
                        else:
                            self.errors.append(
                                f"Invalid {field_name} value '{record_values}'. "
                                f"Valid values are: {values_str}"
                            )


def build(
    index_file: str, output_dir: str, docs_snippets_dir: Optional[str] = None
) -> None:
    """Build the future work widget data.

    Args:
        index_file: Path to the index.json file to process
        output_dir: Path to the output directory to create and save data.json
        docs_snippets_dir: Optional path to docs/snippets directory for
            validation files

    Raises:
        FileNotFoundError: If the index file does not exist
        TypeError: If the index file does not contain a JSON array
        ValueError: If validation errors are encountered in the data
    """
    logging.info(f"Building widget from {index_file}")

    index_path = Path(index_file)
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_file}")

    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError("Index file must contain a JSON array")

    validator = RecordValidator(docs_snippets_dir)
    future_work_items = []

    for item in data:
        validated_item = validator.validate(item)
        if validated_item is not None:
            future_work_items.append(validated_item)

    errors = validator.get_errors()
    if errors:
        for error in errors:
            logging.error(f"Validation error: {error}")
        error_details = "; ".join(errors)
        raise ValueError(
            f"Validation failed with {len(errors)} error(s): {error_details}"
        )

    logging.info(
        f"Found {len(future_work_items)} future-work items out of "
        f"{len(data)} total items"
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data_file = output_path / "data.json"
    with open(data_file, "w", encoding="utf-8") as f:
        json.dump(future_work_items, f, indent=2, ensure_ascii=False)

    logging.info(
        f"Generated data.json with {len(future_work_items)} items in "
        f"{output_dir}"
    )
