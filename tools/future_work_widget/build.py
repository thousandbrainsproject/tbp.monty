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
from typing import Any, Dict, List, Optional


class RecordValidator:
    """Validates and transforms records for the future work widget."""

    COMMA_SEPARATED_FIELDS = ["tags", "owner"]
    MAX_COMMA_SEPARATED_ITEMS = 10
    REQUIRED_FIELDS = ["estimated-scope", "rfc"]
    VALID_ESTIMATED_SCOPE = {"small", "medium", "large", "unknown"}
    VALID_RFC_VALUES = {"required", "optional", "not-required"}
    VALID_STATUS = {"completed", "in-progress"}

    def __init__(self):
        self.errors: List[str] = []

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
        self._validate_estimated_scope(transformed_record)
        self._validate_rfc(transformed_record)
        self._validate_status(transformed_record)

        return transformed_record

    def _validate_required_fields(self, record: Dict[str, Any]) -> None:
        """Validate that all required fields are present and not empty."""
        for field in self.REQUIRED_FIELDS:
            if field not in record:
                self.errors.append(f"Required field '{field}' is missing")
            elif not isinstance(record[field], str) or not record[field].strip():
                self.errors.append(f"Required field '{field}' cannot be empty")

    def _validate_estimated_scope(self, record: Dict[str, Any]) -> None:
        """Validate estimated-scope field."""
        if "estimated-scope" in record:
            value = record["estimated-scope"]
            if not isinstance(value, str) or value not in self.VALID_ESTIMATED_SCOPE:
                valid_values = ", ".join(sorted(self.VALID_ESTIMATED_SCOPE))
                self.errors.append(
                    f"estimated-scope must be one of: {valid_values}. "
                    f"Got: {value}"
                )

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
                f"rfc must be one of: {valid_values} or a valid RFC URL. "
                f"Got: {value}"
            )

    def _validate_status(self, record: Dict[str, Any]) -> None:
        """Validate status field."""
        if "status" in record:
            value = record["status"]
            if not isinstance(value, str) or value not in self.VALID_STATUS:
                valid_values = ", ".join(sorted(self.VALID_STATUS))
                self.errors.append(
                    f"status must be one of: {valid_values}. Got: {value}"
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


def build(index_file: str, output_dir: str) -> None:
    """Build the future work widget data.

    Args:
        index_file: Path to the index.json file to process
        output_dir: Path to the output directory to create and save data.json

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

    validator = RecordValidator()
    future_work_items = []

    for item in data:
        validated_item = validator.validate(item)
        if validated_item is not None:
            future_work_items.append(validated_item)

    errors = validator.get_errors()
    if errors:
        for error in errors:
            logging.error(f"Validation error: {error}")
        raise ValueError(f"Validation failed with {len(errors)} error(s)")

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
