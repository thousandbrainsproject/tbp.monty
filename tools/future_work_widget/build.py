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
from typing import TYPE_CHECKING

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from tools.future_work_widget.validator import (
    ErrorDetail,
    FutureWorkIndex,
    RecordValidator,
)

if TYPE_CHECKING:
    from pathlib import Path


class BuildResult(BaseModel):
    success: bool
    processed_items: int | None = None
    total_items: int | None = None
    errors: list[ErrorDetail] | None = None
    error_message: str | None = None


def build(
    index_file: Path,
    output_dir: Path,
    docs_snippets_dir: Path,
) -> BuildResult:
    """Build the future work widget data.

    Args:
        index_file: Path to the index.json file to process
        output_dir: Path to the output directory to create and save data.json
        docs_snippets_dir: Path to docs/snippets directory for validation files

    Returns:
        BuildResult with validation/build status and details
    """
    try:
        build_result = _validate_params(index_file, output_dir, docs_snippets_dir)
        if build_result is not None:
            return build_result

        with open(index_file, encoding="utf-8") as f:
            raw_data = json.load(f)

        try:
            index = FutureWorkIndex.model_validate(raw_data)
            data = index.root
        except PydanticValidationError as e:
            return BuildResult(
                success=False,
                error_message=f"Invalid index file format: {e}",
            )

        validator = RecordValidator(docs_snippets_dir)
        future_work_items = []
        errors = []

        for item in data:
            if item.get("path1") != "future-work" or "path2" not in item:
                continue

            validation_errors = validator.validate(item)
            if validation_errors:
                errors.extend(validation_errors)
            else:
                validated_item = validator.transform(item)
                future_work_items.append(validated_item)

        if errors:
            return BuildResult(
                success=False,
                processed_items=len(future_work_items),
                total_items=len(data),
                errors=errors,
                error_message=f"Validation failed with {len(errors)} error(s)",
            )

        data_file = output_dir / "data.json"
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(future_work_items, f, indent=2, ensure_ascii=False)

        return BuildResult(
            success=True,
            processed_items=len(future_work_items),
            total_items=len(data),
        )

    except Exception as e:  # noqa: BLE001
        return BuildResult(
            success=False,
            error_message=f"Unexpected error during build: {e}",
        )


def _validate_params(
    index_file: Path,
    output_dir: Path,
    docs_snippets_dir: Path,
) -> BuildResult | None:
    """Validate input paths and setup output directory.

    Returns:
        None if all validations pass, otherwise BuildResult with success=False
    """
    if not index_file.exists():
        return BuildResult(
            success=False,
            error_message=f"Index file not found: {index_file}",
        )

    if not docs_snippets_dir.exists():
        return BuildResult(
            success=False,
            error_message=f"Docs snippets directory not found: {docs_snippets_dir}",
        )

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        return BuildResult(
            success=False,
            error_message=f"Failed to create output directory {output_dir}: {e}",
        )

    return None
