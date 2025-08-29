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
from typing import Optional

from .validator import RecordValidator


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
        f"Generated data.json with {len(future_work_items)} items in {output_dir}"
    )
