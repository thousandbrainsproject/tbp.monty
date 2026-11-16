# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Models for validation for documentation preview bundles."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

SCHEMA_VERSION = 1
"""Current schema version for documentation preview bundles."""

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
"""Pattern accepted for Git commit SHAs in preview bundles."""


class RenderedDocument(BaseModel):
    """A rendered page and its recursively nested child pages."""

    model_config = ConfigDict(strict=True, extra="forbid")

    title: str
    slug: str
    body: str
    hidden: bool
    children: list[RenderedDocument]
    description: str | None = None


class RenderedCategory(BaseModel):
    """A documentation category containing rendered pages."""

    model_config = ConfigDict(strict=True, extra="forbid")

    title: str
    children: list[RenderedDocument]


class PullRequestMetadata(BaseModel):
    """Required pull request metadata checked against trusted job inputs."""

    model_config = ConfigDict(strict=True, extra="forbid")

    number: int = Field(gt=0)
    head_sha: str
    base_ref: str

    @field_validator("number")
    @classmethod
    def validate_number(cls, value: int, info: ValidationInfo) -> int:
        if value != info.context["expected_pr_number"]:
            raise ValueError("does not match the expected pr_number")
        return value

    @field_validator("head_sha")
    @classmethod
    def validate_head_sha(cls, value: str, info: ValidationInfo) -> str:
        if not SHA_RE.fullmatch(value):
            raise ValueError("must be a 40-character SHA")
        if value != info.context["expected_head_sha"]:
            raise ValueError("does not match the expected head_sha")
        return value

    @field_validator("base_ref")
    @classmethod
    def validate_base_ref(cls, value: str, info: ValidationInfo) -> str:
        if value != info.context["expected_base_ref"]:
            raise ValueError("does not match the expected base branch")
        return value


class PreviewBundle(BaseModel):
    """Base model class for all preview bundles."""

    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: int
    repository: str
    pull_request: PullRequestMetadata
    documents: list[RenderedCategory]

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: int) -> int:
        if value != SCHEMA_VERSION:
            raise ValueError(f"must be {SCHEMA_VERSION}, got {value!r}")
        return value

    @field_validator("repository")
    @classmethod
    def validate_repository(cls, value: str, info: ValidationInfo) -> str:
        if value != info.context["expected_repository"]:
            raise ValueError("does not match the current repository")
        return value
