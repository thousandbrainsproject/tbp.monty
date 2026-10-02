# Copyright 2025-2026 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tools.github_readme_sync.colors import BLUE, CYAN, GRAY, RESET, WHITE
from tools.github_readme_sync.hierarchy import INDENTATION_UNIT
from tools.github_readme_sync.md import process_markdown
from tools.github_readme_sync.readme import ReadMe

logger = logging.getLogger(__name__)


@dataclass
class ReadMeItem:
    id: str
    type: Literal["category", "doc"]


def upload(new_hierarchy, file_path: str, rdme: ReadMe):
    logger.info(f"Uploading export folder: {file_path}")
    logger.info(f"URL: https://docs.thousandbrains.org/v{rdme.version}/docs")
    rdme.create_version_if_not_exists()

    to_be_deleted = get_all_categories_docs(rdme)

    for category in new_hierarchy:
        cat_id, created = rdme.create_category_if_not_exists(category["title"])
        logger.info(
            f"\n{BLUE}{category['title'].upper()}{GRAY}{created * ' [created]'}{RESET}"
        )

        # Exact title match (case-sensitive)
        set_do_not_delete(to_be_deleted, category["title"])

        # Recursively process the hierarchy of children
        process_children(
            parent=category,
            cat_id=cat_id,
            file_path=file_path,
            rdme=rdme,
            to_be_deleted=to_be_deleted,
        )

    logger.info("")

    if len(to_be_deleted) > 0:
        # Delete all docs and categories in reverse order
        for item in reversed(to_be_deleted):
            if item.type == "doc":
                rdme.delete_doc(item.id)
            elif item.type == "category":
                rdme.delete_category(item.id)

    # Only expose a stable release after every create, update, and delete
    # operation has completed successfully.
    rdme.make_version_stable()


def upload_rendered(
    new_hierarchy,
    rdme: ReadMe,
):
    """Upload a pre-rendered documentation hierarchy to ReadMe.

    Creates the target ReadMe version if necessary, creates or updates each
    category and document in the supplied hierarchy, and removes resources
    that no longer exist in the hierarchy. Unlike ``upload``, document bodies
    are already rendered and do not require access to the local source files.

    Args:
        new_hierarchy: Pre-rendered documentation hierarchy to upload.
        rdme: ReadMe client configured for the target version.
    """
    logger.info(f"Uploading rendered preview to version: {rdme.version}")
    logger.info(f"URL: https://docs.thousandbrains.org/v{rdme.version}/docs")

    rdme.create_version_if_not_exists()

    to_be_deleted = get_all_categories_docs(rdme)

    for category in new_hierarchy:
        cat_id, created = rdme.create_category_if_not_exists(category["title"])

        logger.info(
            f"\n{BLUE}{category['title'].upper()}{GRAY}{created * ' [created]'}{RESET}"
        )

        set_do_not_delete(
            to_be_deleted,
            category["title"],
        )

        process_rendered_children(
            parent=category,
            cat_id=cat_id,
            rdme=rdme,
            to_be_deleted=to_be_deleted,
        )

    logger.info("")

    if to_be_deleted:
        for item in reversed(to_be_deleted):
            if item.type == "doc":
                rdme.delete_doc(item.id)
            elif item.type == "category":
                rdme.delete_category(item.id)

    rdme.make_version_stable()


def process_rendered_children(
    parent,
    cat_id,
    rdme,
    to_be_deleted: list[ReadMeItem],
    level=0,
    parent_doc_id=None,
):
    """Create or update pre-rendered documents beneath a parent.

    Processes each child document in order, removes documents that still
    exist from the cleanup list, and recursively processes nested children.

    Args:
        parent: Parent category or document containing a ``children`` list.
        cat_id: URI of the ReadMe category containing the documents.
        rdme: ReadMe client used to create or update documents.
        to_be_deleted: Existing ReadMe resources that have not yet been matched
            to resources in the new hierarchy.
        level: Current nesting depth, used when logging document changes.
        parent_doc_id: Optional URI of the parent document in ReadMe.
    """
    for i, doc in enumerate(parent["children"]):
        doc_id, created = rdme.create_or_update_rendered_doc(
            order=i,
            category_id=cat_id,
            doc=doc,
            parent_id=parent_doc_id,
        )

        print_child(
            level=level,
            doc=doc,
            created=created,
        )

        set_do_not_delete(
            to_be_deleted,
            doc["slug"],
        )

        if doc["children"]:
            process_rendered_children(
                parent=doc,
                cat_id=cat_id,
                rdme=rdme,
                to_be_deleted=to_be_deleted,
                level=level + 1,
                parent_doc_id=doc_id,
            )


def process_children(
    parent,
    cat_id,
    file_path,
    rdme,
    to_be_deleted: list[ReadMeItem],
    path_prefix="",
    parent_doc_id=None,
):
    # Process the current level's children
    for i, child in enumerate(parent["children"]):
        doc = load_doc(file_path, f"{path_prefix}{parent['slug']}", child)
        doc_id, created = rdme.create_or_update_doc(
            order=i,
            category_id=cat_id,
            doc=doc,
            parent_id=parent_doc_id,
            file_path=f"{file_path}/{path_prefix}{parent['slug']}",
        )

        print_child(
            level=path_prefix.count("/"),
            doc=doc,
            created=created,
        )
        set_do_not_delete(to_be_deleted, child["slug"])

        # If this child has children, call the function recursively
        if child.get("children"):
            process_children(
                parent=child,
                cat_id=cat_id,
                file_path=file_path,
                rdme=rdme,
                to_be_deleted=to_be_deleted,
                path_prefix=f"{path_prefix}{parent['slug']}/",
                parent_doc_id=doc_id,
            )


def set_do_not_delete(to_be_deleted: list[ReadMeItem], identifier: str):
    for item in to_be_deleted:
        if item.id == identifier:
            to_be_deleted.remove(item)
            return


def get_all_categories_docs(rdme: ReadMe) -> list[ReadMeItem]:
    """Return the ReadMe resources that may need to be removed during cleanup.

    Retrieves every category from the configured ReadMe version and every
    document within those categories. Cleanup only needs each resource's
    identifier and type, so categories are identified by title and documents
    by slug. During an upload, items that still exist in the new hierarchy are removed
    from this list before the remaining items are deleted.

    Args:
        rdme: ReadMe client used to retrieve categories and documents.

    Returns:
        A list of ReadMeItem objects representing all existing categories and
        documents.
    """
    all_categories_and_docs: list[ReadMeItem] = []

    for category in rdme.get_categories():
        all_categories_and_docs.append(
            ReadMeItem(
                id=category["title"],
                type="category",
            )
        )

        # The API returns the category's pages as a flat collection.
        for doc in rdme.get_category_docs(category):
            all_categories_and_docs.append(
                ReadMeItem(
                    id=doc["slug"],
                    type="doc",
                )
            )

    return all_categories_and_docs


def print_child(level: int, doc: dict, created: bool):
    color = CYAN if level else BLUE
    indent = INDENTATION_UNIT * level
    suffix = f"{GRAY}[created]{RESET}" if created else f"{GRAY}[updated]{RESET}"

    logger.info(
        f"{color}{indent}{doc['title']} {WHITE}/{doc['slug']} {GRAY}{suffix}{RESET}"
    )


def load_doc(file_path: str, category_slug: str, child: dict):
    file_path = Path(file_path) / category_slug / f"{child['slug']}.md"
    if not file_path.exists():
        raise ValueError(f"File {file_path} does not exist")

    with file_path.open(encoding="utf-8") as file:
        body = file.read()
        return process_markdown(body, child["slug"])
