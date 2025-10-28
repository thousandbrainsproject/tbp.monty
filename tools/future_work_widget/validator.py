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
from pathlib import Path
from typing import Any

import nh3
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    ValidationInfo,
    field_validator,
)
from pydantic import ValidationError as PydanticValidationError
from typing_extensions import Annotated

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
    root: list[dict[str, Any]]


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
        str | None,
        Field(
            default=None,
            alias="improved-metric",
            description="Metric this work improves",
        ),
    ]
    output_type: Annotated[
        str | None,
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
    def _get_allowed_values(
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

    @field_validator("tags", "skills", mode="before")
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

        field_name, allowed_values = cls._get_allowed_values(info)
        if allowed_values is None:
            return parsed_items

        sanitized_items = []
        for item in parsed_items:
            stripped_item = item.strip()
            if stripped_item not in allowed_values:
                valid_list = ", ".join(sorted(allowed_values))
                raise ValueError(
                    f"Invalid {field_name} value '{stripped_item}'. "
                    f"Valid values are: {valid_list}"
                )
            sanitized_items.append(nh3.clean(stripped_item))

        return sanitized_items

    @field_validator("estimated_scope", "improved_metric", "output_type", "status")
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

        field_name, allowed_values = cls._get_allowed_values(info)
        if allowed_values is None:
            return v

        stripped_value = v.strip()
        if stripped_value not in allowed_values:
            valid_list = ", ".join(sorted(allowed_values))
            field_info = cls.model_fields.get(field_name)
            display_name = (
                field_info.alias if field_info and field_info.alias else field_name
            )
            raise ValueError(
                f"Invalid {display_name} value '{stripped_value}'. "
                f"Valid values are: {valid_list}"
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


class RecordValidator:
    """Validates and transforms records for the future work widget.

    This class loads allowed values from markdown files and passes them
    to Pydantic for validation via the validation context.
    """

    REQUIRED_FIELDS: list[str] = []

    def __init__(self, docs_snippets_dir: Path):
        self.allowed_values: dict[str, list[str]] = {}
        self._load_validation_files(docs_snippets_dir)

    def validate(self, record: dict[str, Any]) -> list[ErrorDetail]:
        """Validate a record using Pydantic with dynamic validation context.

        Args:
            record: The record to validate

        Returns:
            List of error details (empty if validation succeeds)
        """
        file_path = str(record.get("path", "unknown"))

        try:
            validation_context = {"allowed_values": self.allowed_values}
            validated_record = FutureWorkRecord.model_validate(
                record, context=validation_context
            )
            record_dict = validated_record.model_dump()
        except PydanticValidationError as e:
            return self._convert_pydantic_error_to_error_details(e, file_path)

        errors: list[ErrorDetail] = []
        self._validate_required_fields(record_dict, file_path, errors)
        return errors

    def transform(self, record: dict[str, Any]) -> dict[str, Any]:
        """Transform a record using Pydantic after successful validation.

        Args:
            record: The record to transform (must have been validated first)

        Returns:
            Transformed record with parsed fields
        """
        validation_context = {"allowed_values": self.allowed_values}
        validated_record = FutureWorkRecord.model_validate(
            record, context=validation_context
        )
        return validated_record.model_dump(by_alias=True)

    def _create_error_detail(
        self, message: str, file_path: str, field: str
    ) -> ErrorDetail:
        """Create a single error detail object.

        Args:
            message: Error message
            file_path: Path to the file with the error
            field: Field name that has the error

        Returns:
            ErrorDetail object with standardized error information
        """
        return ErrorDetail(
            message=message,
            file=file_path,
            line=1,
            field=field,
            level="error",
            title=f"Validation Error in {Path(file_path).name}",
            annotation_level="failure",
        )

    def _convert_pydantic_error_to_error_details(
        self, exc: PydanticValidationError, file_path: str
    ) -> list[ErrorDetail]:
        error_details = []
        for error in exc.errors():
            field = (
                ".".join(str(loc) for loc in error["loc"])
                if error["loc"]
                else "unknown"
            )
            message = error["msg"]
            error_details.append(self._create_error_detail(message, file_path, field))
        return error_details

    def _validate_required_fields(
        self, record: dict[str, Any], file_path: str, errors: list[ErrorDetail]
    ) -> None:
        """Validate that all required fields are present and not empty."""
        for field in self.REQUIRED_FIELDS:
            if field not in record:
                self._add_error_detail(
                    f"Required field '{field}' is missing", file_path, field, errors
                )
                continue

            if not isinstance(record[field], str):
                self._add_error_detail(
                    f"Required field '{field}' must be a string, "
                    f"got {type(record[field]).__name__}",
                    file_path,
                    field,
                    errors,
                )
                continue

            if not record[field].strip():
                self._add_error_detail(
                    f"Required field '{field}' cannot be empty",
                    file_path,
                    field,
                    errors,
                )
                continue

    def _add_error_detail(
        self, message: str, file_path: str, field: str, errors: list[ErrorDetail]
    ) -> None:
        """Add an error detail to the errors list."""
        errors.append(self._create_error_detail(message, file_path, field))

    def _load_validation_files(self, docs_snippets_dir: Path) -> None:
        """Load validation files from docs_snippets_dir.

        These files define allowed values for fields like tags, skills, etc.
        The values are passed to Pydantic via validation context.

        Args:
            docs_snippets_dir: Path to the snippets directory
        """
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
                self.allowed_values[field_name] = parsed_values
                logger.debug(
                    f"Loaded {len(parsed_values)} allowed values for "
                    f"'{field_name}' from {file_path.name}"
                )
