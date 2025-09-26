# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .validator import RecordValidator


def build(
    index_file: Path,
    output_dir: Path,
    docs_snippets_dir: Path,
) -> dict[str, Any]:
    """Build the future work widget data.

    Args:
        index_file: Path to the index.json file to process
        output_dir: Path to the output directory to create and save data.json
        docs_snippets_dir: Path to docs/snippets directory for validation files

    Returns:
        Dict with keys:
        - success: bool indicating if build was successful
        - processed_items: int number of items processed
        - total_items: int total number of items found
        - errors: list of error dicts with file/line/message info
        - error_message: str summary error message (only if success=False)
    """
    try:
        validation_error = _validate_params(index_file, output_dir, docs_snippets_dir)
        if validation_error is not None:
            return validation_error

        with open(index_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            return {
                "success": False,
                "error_message": "Index file must contain a JSON array",
            }

        validator = RecordValidator(docs_snippets_dir)
        future_work_items = []
        errors = []

        for item in data:
            validated_item, item_errors = validator.validate(item)
            errors.extend(item_errors)
            if validated_item is not None:
                future_work_items.append(validated_item)

        if errors:
            return {
                "success": False,
                "processed_items": len(future_work_items),
                "total_items": len(data),
                "errors": [
                    {
                        "message": error.message,
                        "file": error.file_path,
                        # Hardcoded to 1 because all validation errors
                        # occur at the top of the file in frontmatter.
                        "line": 1,
                        "field": error.field,
                        "level": "error",
                        "title": f"Validation Error in {Path(error.file_path).name}",
                        "annotation_level": "failure",
                    }
                    for error in errors
                ],
                "error_message": f"Validation failed with {len(errors)} error(s)",
            }

        data_file = output_dir / "data.json"
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(future_work_items, f, indent=2, ensure_ascii=False)

        return {
            "success": True,
            "processed_items": len(future_work_items),
            "total_items": len(data),
        }

    except Exception as e:  # noqa: BLE001
        return {
            "success": False,
            "error_message": f"Unexpected error during build: {e}",
        }


def _validate_params(
    index_file: Path,
    output_dir: Path,
    docs_snippets_dir: Path,
) -> dict[str, Any] | None:
    """Validate input paths and setup output directory.

    Returns:
        None if all validations pass, otherwise error dict with success=False
    """
    if not index_file.exists():
        return {
            "success": False,
            "error_message": f"Index file not found: {index_file}",
        }

    if not docs_snippets_dir.exists():
        return {
            "success": False,
            "error_message": f"Docs snippets directory not found: {docs_snippets_dir}",
        }

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        return {
            "success": False,
            "error_message": f"Failed to create output directory {output_dir}: {e}",
        }

    return None
