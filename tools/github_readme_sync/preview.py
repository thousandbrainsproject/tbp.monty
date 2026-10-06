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

from tools.github_readme_sync.hierarchy import check_hierarchy_file
from tools.github_readme_sync.readme import ReadMe
from tools.github_readme_sync.upload import load_doc

SCHEMA_VERSION = 1
"""Current schema version for documentation preview bundles."""


def render_preview(
    folder: str,
    output_file: str,
    *,
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
