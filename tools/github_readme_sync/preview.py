# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from tools.github_readme_sync.hierarchy import check_hierarchy_file
from tools.github_readme_sync.readme import ReadMe
from tools.github_readme_sync.upload import load_doc, upload_rendered

SCHEMA_VERSION = 1
"""Current schema version for documentation preview bundles."""

MAX_BUNDLE_BYTES = 25 * 1024 * 1024
"""Maximum allowed preview bundle size in bytes."""

MAX_CATEGORIES = 200
"""Maximum number of categories allowed in a preview bundle."""

MAX_DOCUMENTS = 2_000
"""Maximum number of documents allowed in a preview bundle."""

MAX_DEPTH = 10
"""Maximum document nesting depth allowed in a preview bundle."""

MAX_TITLE_LENGTH = 300
"""Maximum allowed document or category title length."""

MAX_DESCRIPTION_LENGTH = 2_000
"""Maximum allowed document description length."""

MAX_BODY_LENGTH = 2 * 1024 * 1024
"""Maximum allowed document body length."""

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,199}$")
"""Pattern accepted for document slugs in preview bundles."""

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
"""Pattern accepted for Git commit SHAs in preview bundles."""


def render_preview(
    folder: str,
    output_file: str,
    *,
    repository: str,
    pr_number: int,
    head_sha: str,
    base_ref: str,
) -> None:
    """Render local documentation into a validated preview bundle.

    Loads the documentation hierarchy and source files, renders each document
    body, records the pull request identity metadata, validates the resulting
    bundle, and writes it to disk as JSON.

    Args:
        folder: Directory containing ``hierarchy.md`` and the documentation
            source files.
        output_file: Path where the rendered preview bundle will be written.
        repository: Full repository name associated with the pull request.
        pr_number: Pull request number associated with the preview.
        head_sha: Commit SHA for the pull request head.
        base_ref: Base branch that the preview is expected to target.
    """
    hierarchy = check_hierarchy_file(folder)
    # Markdown rendering does not require a ReadMe version or API access.
    renderer = ReadMe(version="")

    documents = [
        {
            "title": category["title"],
            "children": _render_children(
                parent=category,
                folder=folder,
                renderer=renderer,
            ),
        }
        for category in hierarchy
    ]

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "repository": repository,
        "pull_request": {
            "number": pr_number,
            "head_sha": head_sha,
            "base_ref": base_ref,
        },
        "documents": documents,
    }

    # Validate our own output too. This keeps the producer and consumer on the
    # same schema and catches accidental format drift before the artifact upload.
    validate_preview_bundle(
        bundle,
        expected_repository=repository,
        expected_pr_number=pr_number,
        expected_head_sha=head_sha,
        expected_base_ref=base_ref,
    )

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _render_children(
    parent: dict,
    folder: str,
    renderer: ReadMe,
    path_prefix: str = "",
) -> list[dict]:
    """Render all child documents beneath one hierarchy node.

    Loads each child document, renders its Markdown body, preserves
    preview metadata, and recursively renders any nested child documents.

    Args:
        parent: Category or document whose `children` should be rendered.
        folder: Root directory containing the documentation source files.
        renderer: ReadMe client used only for Markdown rendering.
        path_prefix: Relative path from `folder` to the parent's ancestors.

    Returns:
        The rendered child documents in hierarchy order.
    """
    rendered_children = []
    parent_path = f"{path_prefix}{parent['slug']}"

    for child in parent["children"]:
        doc = load_doc(folder, parent_path, child)

        rendered_doc = {
            "title": doc["title"],
            "slug": doc["slug"],
            "body": renderer.process_markdown(
                doc["body"],
                str(Path(folder) / parent_path),
                doc["slug"],
            ),
            "hidden": bool(doc.get("hidden", False)),
            "children": (
                _render_children(
                    parent=child,
                    folder=folder,
                    renderer=renderer,
                    path_prefix=f"{parent_path}/",
                )
                if child.get("children")
                else []
            ),
        }

        if "description" in doc:
            rendered_doc["description"] = doc["description"]

        rendered_children.append(rendered_doc)

    return rendered_children


def publish_preview(
    bundle_file: str,
    rdme: ReadMe,
    *,
    expected_repository: str,
    expected_pr_number: int,
    expected_head_sha: str,
    expected_base_ref: str,
) -> None:
    """Validate and publish a pre-rendered preview bundle to ReadMe.

    Requires a non-stable preview version, loads the bundle from disk, verifies
    it against trusted pull request metadata, and uploads only the validated
    rendered documentation hierarchy.

    Args:
        bundle_file: Path to the rendered preview JSON bundle.
        rdme: ReadMe client configured for the target preview version.
        expected_repository: Repository name expected in the bundle.
        expected_pr_number: Pull request number expected in the bundle.
        expected_head_sha: Pull request head SHA expected in the bundle.
        expected_base_ref: Base branch expected in the bundle.

    Raises:
        ValueError: If the target is not a preview version or the bundle fails
            validation.
    """
    if not rdme.version_has_suffix():
        raise ValueError("publish-preview requires a non-stable preview version")

    bundle = load_preview_bundle(bundle_file)

    validate_preview_bundle(
        bundle,
        expected_repository=expected_repository,
        expected_pr_number=expected_pr_number,
        expected_head_sha=expected_head_sha,
        expected_base_ref=expected_base_ref,
    )

    upload_rendered(bundle["documents"], rdme)


def load_preview_bundle(bundle_file: str) -> dict[str, Any]:
    """Load a preview bundle from disk with basic size and type checks.

    Args:
        bundle_file: Path to the preview bundle JSON file.

    Returns:
        The parsed preview bundle as a dictionary.

    Raises:
        ValueError: If the bundle exceeds the size limit or is not valid JSON.
        TypeError: If the top-level JSON value is not an object.
    """
    path = Path(bundle_file)
    size = path.stat().st_size

    if size > MAX_BUNDLE_BYTES:
        raise ValueError(
            f"Preview bundle is {size} bytes; maximum is {MAX_BUNDLE_BYTES} bytes"
        )

    try:
        bundle = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Preview bundle is not valid JSON") from exc

    if not isinstance(bundle, dict):
        raise TypeError("Preview bundle must be a JSON object")

    return bundle


def validate_preview_bundle(
    bundle: dict[str, Any],
    *,
    expected_repository: str,
    expected_pr_number: int,
    expected_head_sha: str,
    expected_base_ref: str,
) -> None:
    """Validate a preview bundle before it is uploaded.

    Validates the exact bundle schema, trusted pull request identity metadata,
    category limits, and every rendered document recursively. Document slugs
    must be unique across the entire bundle.

    Args:
        bundle: Parsed preview bundle to validate.
        expected_repository: Repository name trusted by the publishing job.
        expected_pr_number: Pull request number trusted by the publishing job.
        expected_head_sha: Pull request head SHA trusted by the publishing job.
        expected_base_ref: Base branch trusted by the publishing job.

    Raises:
        TypeError: If a bundle field has an invalid type.
        ValueError: If the schema, trusted metadata, or content constraints do
            not match the expected values.
    """
    _require_exact_keys(
        bundle,
        required={
            "schema_version",
            "repository",
            "pull_request",
            "documents",
        },
        path="$",
    )

    schema_version = bundle["schema_version"]

    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise TypeError("$.schema_version must be an integer")

    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"$.schema_version must be {SCHEMA_VERSION}, got {schema_version!r}"
        )

    _require_string(
        bundle["repository"],
        "$.repository",
        max_length=300,
    )

    if bundle["repository"] != expected_repository:
        raise ValueError("$.repository does not match the current repository")

    pull_request = _require_dict(
        bundle["pull_request"],
        "$.pull_request",
    )

    _require_exact_keys(
        pull_request,
        required={
            "number",
            "head_sha",
            "base_ref",
        },
        path="$.pull_request",
    )

    pr_number = pull_request["number"]

    if not isinstance(pr_number, int) or isinstance(pr_number, bool) or pr_number <= 0:
        raise TypeError("$.pull_request.number must be a positive integer")

    if pr_number != expected_pr_number:
        raise ValueError("$.pull_request.number does not match the validated PR")

    head_sha = _require_string(
        pull_request["head_sha"],
        "$.pull_request.head_sha",
        max_length=40,
    )

    if not SHA_RE.fullmatch(head_sha):
        raise ValueError("$.pull_request.head_sha must be a 40-character SHA")

    if head_sha != expected_head_sha:
        raise ValueError("$.pull_request.head_sha does not match the validated PR")

    base_ref = _require_string(
        pull_request["base_ref"],
        "$.pull_request.base_ref",
        max_length=255,
    )

    if base_ref != expected_base_ref:
        raise ValueError(
            "$.pull_request.base_ref does not match the expected base branch"
        )

    categories = bundle["documents"]

    if not isinstance(categories, list):
        raise TypeError("$.documents must be an array")

    if len(categories) > MAX_CATEGORIES:
        raise ValueError(f"$.documents may contain at most {MAX_CATEGORIES} categories")

    state = {
        "document_count": 0,
        "slugs": set(),
    }

    for index, category in enumerate(categories):
        _validate_category(
            category,
            f"$.documents[{index}]",
            state,
        )


def _validate_category(
    category: Any,
    path: str,
    state: dict[str, Any],
) -> None:
    """Validate one category and all documents beneath it.

    Args:
        category: Category value to validate.
        path: JSON-style path used in validation error messages.
        state: Shared validation state containing the document count and slugs
            already encountered in the bundle.

    Raises:
        TypeError: If the category or its children have invalid types.
    """
    category = _require_dict(category, path)

    _require_exact_keys(
        category,
        required={
            "title",
            "children",
        },
        path=path,
    )

    _require_string(
        category["title"],
        f"{path}.title",
        max_length=MAX_TITLE_LENGTH,
    )

    children = category["children"]

    if not isinstance(children, list):
        raise TypeError(f"{path}.children must be an array")

    for index, child in enumerate(children):
        _validate_document(
            child,
            f"{path}.children[{index}]",
            state,
            depth=1,
        )


def _validate_document(
    document: Any,
    path: str,
    state: dict[str, Any],
    *,
    depth: int,
) -> None:
    """Validate one rendered document and its descendants.

    Checks the document schema, field types and limits, slug uniqueness, nesting
    depth, and total document count before recursively validating its children.

    Args:
        document: Rendered document value to validate.
        path: JSON-style path used in validation error messages.
        state: Shared validation state containing the document count and slugs
            already encountered in the bundle.
        depth: Nesting depth of the current document.

    Raises:
        TypeError: If a document field has an invalid type.
        ValueError: If the document violates a schema or content constraint.
    """
    if depth > MAX_DEPTH:
        raise ValueError(f"{path} exceeds the maximum nesting depth of {MAX_DEPTH}")

    document = _require_dict(document, path)

    _require_exact_keys(
        document,
        required={
            "title",
            "slug",
            "body",
            "hidden",
            "children",
        },
        optional={"description"},
        path=path,
    )

    _require_string(
        document["title"],
        f"{path}.title",
        max_length=MAX_TITLE_LENGTH,
    )

    slug = _require_string(
        document["slug"],
        f"{path}.slug",
        max_length=200,
    )

    if not SLUG_RE.fullmatch(slug):
        raise ValueError(
            f"{path}.slug must contain only lowercase letters, digits, and hyphens"
        )

    if slug in state["slugs"]:
        raise ValueError(f"Duplicate document slug in preview bundle: {slug!r}")

    state["slugs"].add(slug)

    body = document["body"]

    if not isinstance(body, str):
        raise TypeError(f"{path}.body must be a string")

    if len(body) > MAX_BODY_LENGTH:
        raise ValueError(
            f"{path}.body exceeds the maximum length of {MAX_BODY_LENGTH} characters"
        )

    if "\x00" in body:
        raise ValueError(f"{path}.body may not contain NUL characters")

    if not isinstance(document["hidden"], bool):
        raise TypeError(f"{path}.hidden must be a boolean")

    if "description" in document:
        _require_string(
            document["description"],
            f"{path}.description",
            max_length=MAX_DESCRIPTION_LENGTH,
            allow_empty=True,
        )

    state["document_count"] += 1

    if state["document_count"] > MAX_DOCUMENTS:
        raise ValueError(
            f"Preview bundle may contain at most {MAX_DOCUMENTS} documents"
        )

    children = document["children"]

    if not isinstance(children, list):
        raise TypeError(f"{path}.children must be an array")

    for index, child in enumerate(children):
        _validate_document(
            child,
            f"{path}.children[{index}]",
            state,
            depth=depth + 1,
        )


def _require_dict(
    value: Any,
    path: str,
) -> dict[str, Any]:
    """Return a value as a dictionary or raise a path-aware error.

    Args:
        value: Value expected to be a dictionary.
        path: JSON-style path identifying the value.

    Returns:
        The validated dictionary.

    Raises:
        TypeError: If ``value`` is not a dictionary.
    """
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be an object")

    return value


def _require_exact_keys(
    value: dict[str, Any],
    *,
    required: set[str],
    path: str,
    optional: set[str] | None = None,
) -> None:
    """Require a dictionary to contain only its declared keys.

    Args:
        value: Dictionary whose keys should be checked.
        required: Keys that must be present.
        path: JSON-style path identifying the dictionary.
        optional: Additional keys that may be present.

    Raises:
        ValueError: If a required key is missing or an unknown key is present.
    """
    optional = optional or set()

    keys = set(value)

    missing = required - keys
    unknown = keys - required - optional

    if missing:
        raise ValueError(f"{path} is missing required keys: {sorted(missing)}")

    if unknown:
        raise ValueError(f"{path} contains unknown keys: {sorted(unknown)}")


def _require_string(
    value: Any,
    path: str,
    *,
    max_length: int,
    allow_empty: bool = False,
) -> str:
    """Validate and return a bounded string value.

    Args:
        value: Value expected to be a string.
        path: JSON-style path identifying the value.
        max_length: Maximum allowed string length.
        allow_empty: Whether an empty or whitespace-only string is permitted.

    Returns:
        The validated string.

    Raises:
        TypeError: If ``value`` is not a string.
        ValueError: If the value is empty when disallowed, exceeds the maximum
            length, or contains a NUL character.
    """
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a string")

    if not allow_empty and not value.strip():
        raise ValueError(f"{path} may not be empty")

    if len(value) > max_length:
        raise ValueError(f"{path} exceeds the maximum length of {max_length}")

    if "\x00" in value:
        raise ValueError(f"{path} may not contain NUL characters")

    return value
