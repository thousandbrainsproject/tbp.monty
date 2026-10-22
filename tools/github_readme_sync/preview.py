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

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
"""Pattern accepted for Git commit SHAs in preview bundles."""


def render_preview(
    folder: Path,
    output_file: Path,
    repository: str,
    pr_number: int,
    head_sha: str,
    base_ref: str,
) -> None:
    """Render local documentation into a validated preview bundle.

    Loads the documentation hierarchy and source files, renders each document
    body, records the pull request identity metadata, validates the resulting
    bundle and writes it to disk as JSON.

    Args:
        folder: Directory containing ``hierarchy.md`` and the documentation
            source files.
        output_file: Path where the rendered preview bundle will be written.
        repository: Full repository name associated with the pull request.
        pr_number: Pull request number associated with the preview.
        head_sha: Commit SHA for the pull request head.
        base_ref: Base branch that the preview is expected to target.
    """
    hierarchy = check_hierarchy_file(str(folder))
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
        bundle=bundle,
        expected_repository=repository,
        expected_pr_number=pr_number,
        expected_head_sha=head_sha,
        expected_base_ref=base_ref,
    )

    with output_file.open(mode="w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False)


def _render_children(
    parent: dict,
    folder: Path,
    renderer: ReadMe,
    path_prefix: Path = Path(),
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
    parent_path = path_prefix / parent["slug"]

    for child in parent["children"]:
        doc = load_doc(str(folder), str(parent_path), child)

        rendered_doc = {
            "title": doc["title"],
            "slug": doc["slug"],
            "body": renderer.process_markdown(
                doc["body"],
                str(folder / parent_path),
                doc["slug"],
            ),
            "hidden": bool(doc.get("hidden", False)),
            "children": (
                _render_children(
                    parent=child,
                    folder=folder,
                    renderer=renderer,
                    path_prefix=parent_path,
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
    bundle_file: Path,
    rdme: ReadMe,
    expected_repository: str,
    expected_pr_number: int,
    expected_head_sha: str,
    expected_base_ref: str,
) -> None:
    """Validate and publish a pre-rendered preview bundle to ReadMe.

    Requires a non-stable preview version, loads the bundle, validates it
    against trusted pull request metadata, and uploads the pre-rendered
    documentation hierarchy.

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
        bundle=bundle,
        expected_repository=expected_repository,
        expected_pr_number=expected_pr_number,
        expected_head_sha=expected_head_sha,
        expected_base_ref=expected_base_ref,
    )

    upload_rendered(bundle["documents"], rdme)


def load_preview_bundle(bundle_file: Path) -> dict[str, Any]:
    """Load a preview bundle.

    Args:
        bundle_file: Path to the preview bundle JSON file.

    Returns:
        The parsed preview bundle as a dictionary.

    Raises:
        ValueError: If the bundle is not valid JSON.
        TypeError: If the top-level JSON value is not an object.
    """
    path = Path(bundle_file)

    try:
        bundle = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Preview bundle is not valid JSON") from exc

    if not isinstance(bundle, dict):
        raise TypeError("Preview bundle must be a JSON object")

    return bundle


def validate_preview_bundle(
    bundle: dict[str, Any],
    expected_repository: str,
    expected_pr_number: int,
    expected_head_sha: str,
    expected_base_ref: str,
) -> None:
    """Validate a preview bundle before it is uploaded.

    Validates the bundle's top-level keys and pull request identity metadata.

    Args:
        bundle: Parsed preview bundle to validate.
        expected_repository: Repository name trusted by the publishing job.
        expected_pr_number: Pull request number trusted by the publishing job.
        expected_head_sha: Pull request head SHA trusted by the publishing job.
        expected_base_ref: Base branch trusted by the publishing job.

    Raises:
        TypeError: If a bundle field has an invalid type.
        ValueError: If the schema or trusted metadata do not match the
            expected values.
    """
    _require_exact_keys(
        value=bundle,
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

    repository = bundle["repository"]

    if not isinstance(repository, str):
        raise TypeError("$.repository must be a string")

    if repository != expected_repository:
        raise ValueError("$.repository does not match the current repository")

    pull_request = bundle["pull_request"]

    _require_exact_keys(
        value=pull_request,
        required={
            "number",
            "head_sha",
            "base_ref",
        },
        path="$.pull_request",
    )

    number = pull_request["number"]

    if not isinstance(number, int) or isinstance(number, bool) or number <= 0:
        raise TypeError("$.pull_request.number must be a positive integer")

    if number != expected_pr_number:
        raise ValueError("$.pull_request.number does not match the expected pr_number")

    head_sha = pull_request["head_sha"]
    if not isinstance(head_sha, str):
        raise TypeError("$.pull_request.head_sha must be a string")

    if not SHA_RE.fullmatch(head_sha):
        raise ValueError("$.pull_request.head_sha must be a 40-character SHA")

    if head_sha != expected_head_sha:
        raise ValueError("$.pull_request.head_sha does not match the expected head_sha")

    base_ref = pull_request["base_ref"]

    if not isinstance(base_ref, str):
        raise TypeError("$.pull_request.base_ref must be a string")

    if base_ref != expected_base_ref:
        raise ValueError(
            "$.pull_request.base_ref does not match the expected base branch"
        )


def _require_exact_keys(
    value: dict[str, Any],
    required: set[str],
    path: str,
) -> None:
    """Require a dictionary to contain only its declared keys.

    Args:
        value: Dictionary whose keys should be checked.
        required: Keys that must be present.
        path: JSON-style path identifying the dictionary.

    Raises:
        ValueError: If a required key is missing or an unknown key is present.
        TypeError: If the `value` arugment isn't a dictionary.
    """
    if not isinstance(value, dict):
        raise TypeError("The value must be an object")

    keys = set(value)

    missing = required - keys
    unknown = keys - required

    if missing:
        raise ValueError(f"{path} is missing required keys: {sorted(missing)}")

    if unknown:
        raise ValueError(f"{path} contains unknown keys: {sorted(unknown)}")
