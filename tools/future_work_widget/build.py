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
from pydantic import ValidationError as PydanticValidationError

from tools.future_work_widget.validator import (
    ErrorDetail,
    FutureWorkIndex,
    find_orphan_values,
    get_snippet_file_path,
    load_allowed_values,
)


class BuildResult(BaseModel):
    success: bool
    future_work_items: int | None = None
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
        error_message = _validate_params(index_file, output_dir, docs_snippets_dir)
        if error_message is not None:
            return BuildResult(success=False, error_message=error_message)

        with open(index_file, encoding="utf-8") as f:
            raw_data = json.load(f)

        total_items = len(raw_data)
        future_work_items = [
            item
            for item in raw_data
            if item.get("path1") == "future-work" and "path2" in item
        ]

        allowed_values = load_allowed_values(docs_snippets_dir)

        try:
            index = FutureWorkIndex.model_validate(
                future_work_items, context={"allowed_values": allowed_values}
            )
        except PydanticValidationError as e:
            return _return_error_result(e, future_work_items, total_items)

        orphan_errors = _check_orphan_values(
            index.root, allowed_values, docs_snippets_dir
        )
        if orphan_errors:
            return BuildResult(
                success=False,
                future_work_items=len(index.root),
                total_items=total_items,
                errors=orphan_errors,
                error_message=f"Found {len(orphan_errors)} orphan value error(s)",
            )

        data_file = output_dir / "data.json"
        with open(data_file, "w", encoding="utf-8") as f:
            f.write(index.model_dump_json(indent=2, by_alias=True))

        return BuildResult(
            success=True,
            future_work_items=len(index.root),
            total_items=total_items,
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
) -> str | None:
    """Validate input paths and setup output directory.

    Returns:
        None if all validations pass, otherwise error message string
    """
    if not index_file.exists():
        return f"Index file not found: {index_file}"

    if not docs_snippets_dir.exists():
        return f"Docs snippets directory not found: {docs_snippets_dir}"

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        return f"Failed to create output directory {output_dir}: {e}"

    return None


def _check_orphan_values(
    records: list,
    allowed_values: dict[str, list[str]],
    docs_snippets_dir: Path,
) -> list[ErrorDetail]:
    """Check for orphan values and return error details.

    Args:
        records: List of validated FutureWorkRecord objects
        allowed_values: Dict mapping field names to their allowed values
        docs_snippets_dir: Path to the docs/snippets directory

    Returns:
        List of ErrorDetail objects for any orphan values found
    """
    orphans = find_orphan_values(records, allowed_values)
    errors = []

    for field_name, orphan_set in orphans.items():
        snippet_path = get_snippet_file_path(docs_snippets_dir, field_name)
        display_name = field_name.replace("_", "-")

        for orphan_value in sorted(orphan_set):
            errors.append(
                ErrorDetail(
                    message=(
                        f"Orphan {display_name} value '{orphan_value}' is defined "
                        f"but not used in any future work item. "
                        f"Remove it from {Path(snippet_path).name} or use it in a "
                        f"future work file."
                    ),
                    file=snippet_path,
                    line=1,
                    field=display_name,
                    level="error",
                    title=f"Orphan {display_name} value in {Path(snippet_path).name}",
                    annotation_level="failure",
                )
            )

    return errors


def _return_error_result(
    exc: PydanticValidationError,
    filtered_data: list[dict],
    total_items: int,
) -> BuildResult:
    errors = []
    for error in exc.errors():
        loc = error["loc"]
        if loc and isinstance(loc[0], int):
            record_index = loc[0]
            file_path = filtered_data[record_index].get("path", "unknown")
        else:
            file_path = "unknown"

        field_parts = [str(item) for item in loc[1:]]
        field = ".".join(field_parts) if len(loc) > 1 else "unknown"

        title = (
            f"Validation Error in {Path(file_path).name}"
            if file_path != "unknown"
            else "Validation Error"
        )

        errors.append(
            ErrorDetail(
                message=error["msg"],
                file=file_path,
                line=1,
                field=field,
                level="error",
                title=title,
                annotation_level="failure",
            )
        )

    return BuildResult(
        success=False,
        future_work_items=0,
        total_items=total_items,
        errors=errors,
        error_message=f"Validation failed with {len(errors)} error(s)",
    )
