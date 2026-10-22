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
from pathlib import Path
from typing import Any

from tools.github_readme_sync.hierarchy import check_hierarchy_file
from tools.github_readme_sync.readme import ReadMe
from tools.github_readme_sync.upload import load_doc, upload_rendered

SCHEMA_VERSION = 1
"""Current schema version for documentation preview bundles."""


def render_preview(
    folder: Path,
    output_file: Path,
    repository: str,
    pr_number: int,
    head_sha: str,
    base_ref: str,
) -> None:
    """Render local documentation into a preview bundle.

    Loads the documentation hierarchy and source files, renders each document
    body, records the pull request identity metadata, and writes the bundle
    to disk as JSON.

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
    expected_repository: str,  # noqa: ARG001
    expected_pr_number: int,  # noqa: ARG001
    expected_head_sha: str,  # noqa: ARG001
    expected_base_ref: str,  # noqa: ARG001
) -> None:
    """Publish a pre-rendered preview bundle to ReadMe.

    Requires a non-stable preview version, loads the bundle, and uploads only
    the pre-rendered documentation hierarchy. Validation work using these
    metadata fields will be added in the next PR.

    Args:
        bundle_file: Path to the rendered preview JSON bundle.
        rdme: ReadMe client configured for the target preview version.
        expected_repository: Repository name expected in the bundle.
        expected_pr_number: Pull request number expected in the bundle.
        expected_head_sha: Pull request head SHA expected in the bundle.
        expected_base_ref: Base branch expected in the bundle.

    Raises:
        ValueError: If the target is not a preview version.
    """
    if not rdme.version_has_suffix():
        raise ValueError("publish-preview requires a non-stable preview version")

    bundle = load_preview_bundle(bundle_file)

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
