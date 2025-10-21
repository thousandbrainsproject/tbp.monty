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

from pydantic import BaseModel

from .validator import RecordValidator


class ErrorDetail(BaseModel):
    message: str
    file: str
    line: int
    field: str
    level: str
    title: str
    annotation_level: str


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
        validation_error = _validate_params(index_file, output_dir, docs_snippets_dir)
        if validation_error is not None:
            return validation_error

        with open(index_file, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            return BuildResult(
                success=False,
                error_message="Index file must contain a JSON array",
            )

        validator = RecordValidator(docs_snippets_dir)
        future_work_items = []
        errors = []

        for item in data:
            validated_item, item_errors = validator.validate(item)
            errors.extend(item_errors)
            if validated_item is not None:
                future_work_items.append(validated_item)

        if errors:
            return BuildResult(
                success=False,
                processed_items=len(future_work_items),
                total_items=len(data),
                errors=[
                    ErrorDetail(
                        message=error.message,
                        file=error.file_path,
                        line=1,
                        field=error.field,
                        level="error",
                        title=f"Validation Error in {Path(error.file_path).name}",
                        annotation_level="failure",
                    )
                    for error in errors
                ],
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
