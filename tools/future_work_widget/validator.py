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
from typing import TYPE_CHECKING, Any

import nh3
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic import ValidationError as PydanticValidationError
from typing_extensions import Annotated

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class FieldError(BaseModel):
    """Represents a validation error for a specific field in a record."""

    message: str
    file_path: str
    field: str

    model_config = ConfigDict(extra="forbid")


class FutureWorkRecord(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        str_strip_whitespace=True,
    )

    path: Annotated[
        str,
        Field(description="Path to the future work markdown file"),
    ]
    path1: Annotated[
        str | None,
        Field(default=None, description="First component of the path (directory)"),
    ]
    path2: Annotated[
        str | None,
        Field(default=None, description="Second component of the path (filename)"),
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

    @field_validator("tags", "skills", mode="before")
    @classmethod
    def parse_comma_separated(cls, v: Any) -> list[str] | None:
        if v is None:
            return None
        if isinstance(v, list):
            items = v
        elif isinstance(v, str):
            items = [item.strip() for item in v.split(",")]
        else:
            return None

        max_items = 10
        if len(items) > max_items:
            raise ValueError(
                f"Cannot have more than {max_items} items. Got {len(items)} items"
            )

        return items

    @field_validator("tags", "skills", mode="after")
    @classmethod
    def validate_allowed_values(
        cls, v: list[str] | None, info: ValidationInfo
    ) -> list[str] | None:
        if v is None:
            return None

        if info.context is None:
            return v

        field_name = info.field_name
        allowed_values = info.context.get("allowed_values", {}).get(field_name)

        if allowed_values is None:
            return v

        sanitized_items = []
        for item in v:
            sanitized = nh3.clean(item).strip()
            if sanitized not in allowed_values:
                valid_list = ", ".join(sorted(allowed_values))
                raise ValueError(
                    f"Invalid {field_name} value '{sanitized}'. "
                    f"Valid values are: {valid_list}"
                )
            sanitized_items.append(sanitized)

        return sanitized_items

    @field_validator("contributor", mode="before")
    @classmethod
    def parse_contributor(cls, v: Any) -> list[str] | None:
        if v is None:
            return None
        if isinstance(v, list):
            contributors = v
        elif isinstance(v, str):
            contributors = [item.strip() for item in v.split(",")]
        else:
            return None

        max_contributors = 10
        if len(contributors) > max_contributors:
            raise ValueError(
                f"Cannot have more than {max_contributors} items. "
                f"Got {len(contributors)} items"
            )

        github_pattern = r"[a-zA-Z0-9][a-zA-Z0-9-]{0,38}"
        for contributor in contributors:
            sanitized = nh3.clean(contributor).strip()
            if not re.fullmatch(github_pattern, sanitized):
                raise ValueError(
                    f"Invalid contributor username '{sanitized}'. "
                    f"Must be valid GitHub username (1-39 characters, "
                    f"alphanumeric and hyphens, cannot start with hyphen)"
                )
        return [nh3.clean(c).strip() for c in contributors]

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

    @model_validator(mode="after")
    def validate_extra_fields(self, info: ValidationInfo) -> FutureWorkRecord:
        if info.context is None:
            return self

        allowed_values = info.context.get("allowed_values", {})
        if not allowed_values:
            return self

        model_dict = self.model_dump()
        defined_fields = {
            "path",
            "path1",
            "path2",
            "tags",
            "skills",
            "rfc",
            "contributor",
        }

        for field_name, field_value in model_dict.items():
            if field_name in defined_fields:
                continue

            if field_value is None:
                continue

            field_allowed_values = allowed_values.get(field_name)
            if field_allowed_values is None:
                continue

            values_to_check = (
                field_value if isinstance(field_value, list) else [field_value]
            )

            for value in values_to_check:
                if value is None:
                    continue

                sanitized = nh3.clean(str(value)).strip()
                if sanitized not in field_allowed_values:
                    valid_list = ", ".join(sorted(field_allowed_values))
                    raise ValueError(
                        f"Invalid {field_name} value '{sanitized}'. "
                        f"Valid values are: {valid_list}"
                    )

        return self


class RecordValidator:
    """Validates and transforms records for the future work widget.

    This class loads allowed values from markdown files and passes them
    to Pydantic for validation via the validation context.
    """

    REQUIRED_FIELDS: list[str] = []
    MAX_COMMA_SEPARATED_ITEMS = 10

    def __init__(self, docs_snippets_dir: Path):
        self.allowed_values: dict[str, list[str]] = {}
        self._load_validation_files(docs_snippets_dir)

    def validate(
        self, record: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, list[ValidationError]]:
        """Validate a record using Pydantic with dynamic validation context.

        Args:
            record: The record to validate

        Returns:
            Tuple of (record or None if there are errors, list of validation errors)
        """
        if record.get("path1") != "future-work" or "path2" not in record:
            return None, []

        errors: list[ValidationError] = []

        if "path" not in record:
            errors.append(
                ValidationError(
                    message="Record is missing required 'path' field",
                    file_path="unknown",
                    field="path",
                )
            )
            return None, errors

        file_path = str(record["path"])

        try:
            validation_context = {"allowed_values": self.allowed_values}
            validated_record = FutureWorkRecord.model_validate(
                record, context=validation_context
            )
            record_dict = validated_record.model_dump()
        except PydanticValidationError as e:
            errors.extend(
                self._convert_pydantic_error_to_validation_errors(e, file_path)
            )
            return None, errors

        self._validate_required_fields(record_dict, file_path, errors)
        return (None if errors else record_dict, errors)

    def _convert_pydantic_error_to_validation_errors(
        self, exc: PydanticValidationError, file_path: str
    ) -> list[ValidationError]:
        validation_errors = []
        for error in exc.errors():
            field = (
                ".".join(str(loc) for loc in error["loc"])
                if error["loc"]
                else "unknown"
            )
            message = error["msg"]
            validation_errors.append(
                ValidationError(
                    message=message,
                    file_path=file_path,
                    field=field,
                )
            )
        return validation_errors

    def _validate_required_fields(
        self, record: dict[str, Any], file_path: str, errors: list[ValidationError]
    ) -> None:
        """Validate that all required fields are present and not empty."""
        for field in self.REQUIRED_FIELDS:
            if field not in record:
                self._add_validation_error(
                    f"Required field '{field}' is missing", file_path, field, errors
                )
                continue

            if not isinstance(record[field], str):
                self._add_validation_error(
                    f"Required field '{field}' must be a string, "
                    f"got {type(record[field]).__name__}",
                    file_path,
                    field,
                    errors,
                )
                continue

            if not record[field].strip():
                self._add_validation_error(
                    f"Required field '{field}' cannot be empty",
                    file_path,
                    field,
                    errors,
                )
                continue

    def _add_validation_error(
        self, message: str, file_path: str, field: str, errors: list[ValidationError]
    ) -> None:
        """Add a ValidationError to the errors list."""
        errors.append(
            ValidationError(message=message, file_path=file_path, field=field)
        )

    def _load_validation_files(self, docs_snippets_dir: Path) -> None:
        """Load validation files from docs/snippets directory.

        These files define allowed values for fields like tags, skills, etc.
        The values are passed to Pydantic via validation context.

        Args:
            docs_snippets_dir: Path to the docs/snippets directory
        """
        future_work_files = list(docs_snippets_dir.glob("future-work-*.md"))

        for file_path in future_work_files:
            field_name = file_path.stem.replace("future-work-", "")

            with open(file_path, encoding="utf-8") as f:
                content = f.read().strip()

            simple_values = []
            for raw_item in content.split("`"):
                clean_item = nh3.clean(raw_item).strip()
                if clean_item:
                    simple_values.append(clean_item)

            if simple_values:
                self.allowed_values[field_name] = simple_values
                logger.debug(
                    f"Loaded {len(simple_values)} allowed values for "
                    f"'{field_name}' from {file_path.name}"
                )
