# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin

import nh3
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    ValidationInfo,
    field_validator,
)
from typing_extensions import Annotated

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

MAX_COMMA_SEPARATED_ITEMS = 10


class ErrorDetail(BaseModel):
    message: str
    file: str
    line: int
    field: str
    level: str
    title: str
    annotation_level: str


class FutureWorkIndex(RootModel):
    root: list[FutureWorkRecord]


class FutureWorkRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    path: Annotated[
        str,
        Field(min_length=1, description="Path to the future work markdown file"),
    ]
    path1: Annotated[
        str | None,
        Field(default=None, description="First component of the path (directory)"),
    ]
    path2: Annotated[
        str | None,
        Field(default=None, description="Second component of the path (filename)"),
    ]
    title: Annotated[
        str | None,
        Field(default=None, description="Title of the future work item"),
    ]
    description: Annotated[
        str | None,
        Field(default=None, description="Description of the future work item"),
    ]
    slug: Annotated[
        str | None,
        Field(default=None, description="URL slug for the future work item"),
    ]
    text: Annotated[
        str | None,
        Field(default=None, description="Text content of the future work item"),
    ]
    estimated_scope: Annotated[
        str | None,
        Field(
            default=None,
            alias="estimated-scope",
            description="Estimated scope of the work",
        ),
    ]
    improved_metric: Annotated[
        list[str] | None,
        Field(
            default=None,
            alias="improved-metric",
            description="Metric this work improves",
        ),
    ]
    output_type: Annotated[
        list[str] | None,
        Field(
            default=None,
            alias="output-type",
            description="Type of output produced",
        ),
    ]
    status: Annotated[
        str | None,
        Field(default=None, description="Current status of the work"),
    ]
    tags: Annotated[
        list[str] | None,
        Field(
            default=None,
            description="Categorization tags for the work item",
            max_length=10,
        ),
    ]
    skills: Annotated[
        list[str] | None,
        Field(
            default=None,
            description="Required skills for completing this work",
            max_length=10,
        ),
    ]
    rfc: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Related RFC reference or status (required/optional/not-required)"
            ),
        ),
    ]
    contributor: Annotated[
        list[str] | None,
        Field(
            default=None,
            description="GitHub usernames of contributors",
            max_length=10,
        ),
    ]

    @classmethod
    def _allowed_values(
        cls, info: ValidationInfo
    ) -> tuple[str | None, list[str] | None]:
        """Extract allowed values from validation context.

        Args:
            info: Validation context containing field name and allowed values

        Returns:
            Tuple of (field_name, allowed_values) or (None, None) if not available
        """
        if info.context is None:
            return None, None

        field_name = info.field_name
        allowed_values = info.context.get("allowed_values", {}).get(field_name)
        return field_name, allowed_values

    @classmethod
    def _parse_comma_separated_list(
        cls, v: Any, max_items: int = MAX_COMMA_SEPARATED_ITEMS
    ) -> list[str] | None:
        """Parse and validate comma-separated string into list.

        Args:
            v: Raw field value (string, list, or None)
            max_items: Maximum number of items allowed

        Returns:
            Parsed list of strings, or None if input is None

        Raises:
            ValueError: If list exceeds max length
        """
        if v is None:
            return None
        if isinstance(v, list):
            parsed_items = v
        elif isinstance(v, str):
            parsed_items = [item.strip() for item in v.split(",")]
        else:
            return None

        if len(parsed_items) > max_items:
            msg = f"Cannot have more than {max_items} items. "
            msg += f"Got {len(parsed_items)} items"
            raise ValueError(msg)

        return parsed_items

    @field_validator("tags", "skills", "output_type", "improved_metric", mode="before")
    @classmethod
    def validate_comma_separated_list(
        cls, v: Any, info: ValidationInfo
    ) -> list[str] | None:
        """Parse comma-separated strings and validate against allowed values.

        Runs before Pydantic type coercion to transform string input into
        lists. Enforces max length and validates items against allowed values
        from validation context if available.

        Args:
            v: Raw field value (string, list, or None)
            info: Validation context containing allowed values

        Returns:
            Parsed and validated list of strings, or None if input is None

        Raises:
            ValueError: If list exceeds max length or contains invalid values
        """
        parsed_items = cls._parse_comma_separated_list(v)
        if parsed_items is None:
            return None

        field_name, allowed_values = cls._allowed_values(info)
        if allowed_values is None:
            return parsed_items

        sanitized_items = []
        for item in parsed_items:
            stripped_item = item.strip()
            if stripped_item not in allowed_values:
                valid_list = ", ".join(sorted(allowed_values))
                field_info = cls.model_fields.get(field_name)
                display_name = (
                    field_info.alias if field_info and field_info.alias else field_name
                )
                snippet_file = (
                    f"docs/snippets/future-work-{field_name.replace('_', '-')}.md"
                )
                raise ValueError(
                    f"Invalid {display_name} value '{stripped_item}'. "
                    f"Valid values are: {valid_list}. "
                    f"To add a new value, edit {snippet_file}"
                )
            sanitized_items.append(nh3.clean(stripped_item))

        return sanitized_items

    @field_validator("estimated_scope", "status")
    @classmethod
    def validate_single_value_field(
        cls, v: str | None, info: ValidationInfo
    ) -> str | None:
        """Validate single-value fields against allowed values.

        Checks field values against allowed values from validation context
        if available.

        Args:
            v: Field value
            info: Validation context containing allowed values

        Returns:
            The validated value, or None if input is None

        Raises:
            ValueError: If value is not in allowed values
        """
        if v is None:
            return None

        field_name, allowed_values = cls._allowed_values(info)
        if allowed_values is None:
            return v

        stripped_value = v.strip()
        if stripped_value not in allowed_values:
            valid_list = ", ".join(sorted(allowed_values))
            field_info = cls.model_fields.get(field_name)
            display_name = (
                field_info.alias if field_info and field_info.alias else field_name
            )
            snippet_file = (
                f"docs/snippets/future-work-{field_name.replace('_', '-')}.md"
            )
            raise ValueError(
                f"Invalid {display_name} value '{stripped_value}'. "
                f"Valid values are: {valid_list}. "
                f"To add a new value, edit {snippet_file}"
            )

        return nh3.clean(stripped_value)

    @field_validator("contributor", mode="before")
    @classmethod
    def validate_contributor(cls, v: Any) -> list[str] | None:
        """Parse and validate GitHub contributor usernames.

        Runs before Pydantic type coercion to transform comma-separated
        strings into lists. Validates each username against GitHub's
        username format requirements.

        Args:
            v: Raw field value (string, list, or None)

        Returns:
            Parsed and validated list of GitHub usernames, or None if
            input is None

        Raises:
            ValueError: If list exceeds max length or username is invalid
        """
        contributors = cls._parse_comma_separated_list(v)
        if contributors is None:
            return None

        github_pattern = r"[a-zA-Z0-9][a-zA-Z0-9-]{0,38}"
        validated_contributors = []
        for contributor in contributors:
            stripped_contributor = contributor.strip()
            if not re.fullmatch(github_pattern, stripped_contributor):
                raise ValueError(
                    f"Invalid contributor username '{stripped_contributor}'. "
                    f"Must be valid GitHub username (1-39 characters, "
                    f"alphanumeric and hyphens, cannot start with hyphen)"
                )
            validated_contributors.append(stripped_contributor)
        return validated_contributors

    @field_validator("rfc")
    @classmethod
    def validate_rfc(cls, v: str | None) -> str | None:
        if v is None:
            return None

        value = v.strip()
        rfc_patterns = [
            r"https://github\.com/thousandbrainsproject/tbp\.monty/.*",
            r"required",
            r"optional",
            r"not-required",
        ]

        if not any(re.fullmatch(pattern, value) for pattern in rfc_patterns):
            valid_options = "a GitHub URL, 'required', 'optional', or 'not-required'"
            raise ValueError(f"Invalid rfc value '{value}'. Must be {valid_options}")

        return value


def load_allowed_values(docs_snippets_dir: Path) -> dict[str, list[str]]:
    allowed_values: dict[str, list[str]] = {}
    future_work_files = list(docs_snippets_dir.glob("future-work-*.md"))

    for file_path in future_work_files:
        field_name = file_path.stem.replace("future-work-", "").replace("-", "_")

        with open(file_path, encoding="utf-8") as f:
            content = f.read().strip()

        parsed_values = []
        for raw_item in content.split("`"):
            clean_item = nh3.clean(raw_item).strip()
            if clean_item:
                parsed_values.append(clean_item)

        if parsed_values:
            allowed_values[field_name] = parsed_values
            logger.debug(
                f"Loaded {len(parsed_values)} allowed values for "
                f"'{field_name}' from {file_path.name}"
            )

    return allowed_values


def _get_list_fields() -> set[str]:
    """Get field names from FutureWorkRecord that are list types.

    Returns:
        Set of field names that have list[str] type annotation
    """
    list_fields = set()
    for field_name, field_info in FutureWorkRecord.model_fields.items():
        annotation = field_info.annotation
        origin = get_origin(annotation)
        if origin is list:
            list_fields.add(field_name)
        elif origin is Union:
            for arg in get_args(annotation):
                if get_origin(arg) is list:
                    list_fields.add(field_name)
                    break
    return list_fields


def find_orphan_values(
    records: list[FutureWorkRecord],
    allowed_values: dict[str, list[str]],
) -> dict[str, set[str]]:
    """Find values defined in allowed_values that are not used in any record.

    Args:
        records: List of validated FutureWorkRecord objects
        allowed_values: Dict mapping field names to their allowed values

    Returns:
        Dict mapping field names to sets of orphan values (defined but unused)
    """
    list_fields = _get_list_fields()
    checkable_fields = set(allowed_values.keys()) & list_fields

    used_values: dict[str, set[str]] = {field: set() for field in checkable_fields}

    for record in records:
        for field_name in checkable_fields:
            field_value = getattr(record, field_name, None)
            if field_value:
                used_values[field_name].update(field_value)

    orphans: dict[str, set[str]] = {}
    for field_name in checkable_fields:
        defined_values = set(allowed_values.get(field_name, []))
        unused = defined_values - used_values[field_name]
        if unused:
            orphans[field_name] = unused

    return orphans


def get_snippet_file_path(
    docs_snippets_dir: Path,
    field_name: str,
) -> str:
    """Get the path to the snippet file for a given field.

    Args:
        docs_snippets_dir: Path to the docs/snippets directory
        field_name: Field name (e.g., "skills", "output_type")

    Returns:
        String path to the snippet file
    """
    display_name = field_name.replace("_", "-")
    return str(docs_snippets_dir / f"future-work-{display_name}.md")
