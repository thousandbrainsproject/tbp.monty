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
import logging
from pathlib import Path
from typing import Any, Dict

from .validator import RecordValidator


def build(
    index_file: str,
    output_dir: str,
    docs_snippets_dir: str | None = None,
) -> Dict[str, Any]:
    """Build the future work widget data.

    Args:
        index_file: Path to the index.json file to process
        output_dir: Path to the output directory to create and save data.json
        docs_snippets_dir: Optional path to docs/snippets directory for
            validation files

    Returns:
        Dict with keys:
        - success: bool indicating if build was successful
        - processed_items: int number of items processed
        - total_items: int total number of items found
        - errors: list of error dicts with file/line/message info
        - error_message: str summary error message (only if success=False)
    """
    try:
        logging.info(f"Building widget from {index_file}")

        index_path = Path(index_file)
        if not index_path.exists():
            return {
                "success": False,
                "error_message": f"Index file not found: {index_file}",
            }

        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            return {
                "success": False,
                "error_message": "Index file must contain a JSON array",
            }

        validator = RecordValidator(docs_snippets_dir)
        future_work_items = []

        for item in data:
            validated_item = validator.validate(item)
            if validated_item is not None:
                future_work_items.append(validated_item)

        errors = validator.get_errors()
        if errors:
            return {
                "success": False,
                "processed_items": len(future_work_items),
                "total_items": len(data),
                "errors": [
                    {
                        "message": error.message,
                        "file": error.file_path,
                        "line": error.line_number,
                        "field": error.field,
                        "level": "error",
                        "title": f"Validation Error in {Path(error.file_path).name}",
                        "annotation_level": "failure",
                    }
                    for error in errors
                ],
                "error_message": f"Validation failed with {len(errors)} error(s)",
            }

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
            f"Generated data.json with {len(future_work_items)} items in {output_dir}"
        )

        return {
            "success": True,
            "processed_items": len(future_work_items),
            "total_items": len(data),
            "errors": [],
        }

    except (OSError, PermissionError) as e:
        return {
            "success": False,
            "error_message": f"File system error: {e}",
        }
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error_message": f"Invalid JSON in index file: {e}",
        }
    except TypeError as e:
        return {
            "success": False,
            "error_message": f"Data serialization error: {e}",
        }
