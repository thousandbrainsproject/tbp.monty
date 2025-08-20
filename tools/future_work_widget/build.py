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
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class RecordValidator:
    """Validates and transforms records for the future work widget."""

    COMMA_SEPARATED_FIELDS = ["tags", "owner", "skills"]
    MAX_COMMA_SEPARATED_ITEMS = 10
    REQUIRED_FIELDS = ["estimated-scope", "rfc"]
    VALID_RFC_VALUES = {"required", "optional", "not-required"}

    def __init__(self, docs_snippets_dir: Optional[str] = None):
        self.errors: List[str] = []
        self.validation_sets: Dict[str, Set[str]] = {}
        if docs_snippets_dir:
            self._load_validation_files(docs_snippets_dir)

    def validate(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate and transform a record.

        Args:
            record: The record to validate and transform

        Returns:
            The transformed record if valid, None if invalid
        """
        if not isinstance(record, dict):
            self.errors.append("Record must be a dictionary")
            return None

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
        self._validate_rfc(transformed_record)
        self._validate_field_values(transformed_record)

        return transformed_record

    def _validate_required_fields(self, record: Dict[str, Any]) -> None:
        """Validate that all required fields are present and not empty."""
        for field in self.REQUIRED_FIELDS:
            if field not in record:
                self.errors.append(f"Required field '{field}' is missing")
            elif not isinstance(record[field], str) or not record[field].strip():
                self.errors.append(f"Required field '{field}' cannot be empty")

    def _validate_rfc(self, record: Dict[str, Any]) -> None:
        """Validate rfc field."""
        if "rfc" in record:
            value = record["rfc"]
            if not isinstance(value, str):
                self.errors.append("rfc must be a string")
                return

            if value in self.VALID_RFC_VALUES:
                return

            if self._is_valid_rfc_url(value):
                return

            valid_values = ", ".join(sorted(self.VALID_RFC_VALUES))
            self.errors.append(
                f"rfc must be one of: {valid_values} or a valid RFC URL. Got: {value}"
            )

    def _is_valid_rfc_url(self, url: str) -> bool:
        """Check if the URL is a valid RFC URL.

        Returns:
            True if the URL is a valid RFC URL, False otherwise
        """
        github_patterns = [
            "github.com/thousandbrainsproject/tbp.monty/pull/",
            "https://github.com/thousandbrainsproject/tbp.monty/pull/",
        ]
        return any(url.startswith(pattern) for pattern in github_patterns)

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

                valid_values = set()
                for raw_item in content.split("`"):
                    clean_item = raw_item.strip()
                    if clean_item:
                        valid_values.add(clean_item)

                if valid_values:
                    self.validation_sets[field_name] = valid_values
                    logging.info(
                        f"Loaded {len(valid_values)} valid values for '{field_name}' "
                        f"from {file_path.name}"
                    )

            except (OSError, UnicodeDecodeError):
                logging.exception(f"Failed to load validation file {file_path}")

    def _validate_field_values(self, record: Dict[str, Any]) -> None:
        """Validate field values against loaded validation sets.

        Args:
            record: The record to validate
        """
        for field_name, valid_values in self.validation_sets.items():
            if field_name in record:
                record_values = record[field_name]

                if isinstance(record_values, list):
                    for value in record_values:
                        if value not in valid_values:
                            sorted_valid = sorted(valid_values)
                            self.errors.append(
                                f"Invalid {field_name} value '{value}'. "
                                f"Valid values are: {', '.join(sorted_valid)}"
                            )
                elif isinstance(record_values, str):
                    if record_values not in valid_values:
                        sorted_valid = sorted(valid_values)
                        self.errors.append(
                            f"Invalid {field_name} value '{record_values}'. "
                            f"Valid values are: {', '.join(sorted_valid)}"
                        )

        # Handle estimated-scope field specifically (required field)
        if "estimated-scope" in record:
            value = record["estimated-scope"]
            if "estimated-scope" in self.validation_sets:
                # Validate against loaded validation set
                valid_scopes = self.validation_sets["estimated-scope"]
                if not isinstance(value, str) or value not in valid_scopes:
                    sorted_valid = sorted(valid_scopes)
                    self.errors.append(
                        f"estimated-scope must be one of: {', '.join(sorted_valid)}. "
                        f"Got: {value}"
                    )
            else:
                # Fallback to hardcoded values if snippet file not available
                fallback_scopes = {"small", "medium", "large", "unknown"}
                if not isinstance(value, str) or value not in fallback_scopes:
                    sorted_valid = sorted(fallback_scopes)
                    self.errors.append(
                        f"estimated-scope must be one of: {', '.join(sorted_valid)}. "
                        f"Got: {value}"
                    )

        # Handle status field specifically
        if "status" in record:
            value = record["status"]
            if "status" in self.validation_sets:
                # Validate against loaded validation set
                valid_statuses = self.validation_sets["status"]
                if not isinstance(value, str) or value not in valid_statuses:
                    sorted_valid = sorted(valid_statuses)
                    self.errors.append(
                        f"status must be one of: {', '.join(sorted_valid)}. "
                        f"Got: {value}"
                    )
            else:
                # Fallback to hardcoded values if snippet file not available
                fallback_statuses = {"completed", "in-progress"}
                if not isinstance(value, str) or value not in fallback_statuses:
                    sorted_valid = sorted(fallback_statuses)
                    self.errors.append(
                        f"status must be one of: {', '.join(sorted_valid)}. "
                        f"Got: {value}"
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
