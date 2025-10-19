# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import logging
import os

from tools.github_readme_sync.colors import BLUE, CYAN, GRAY, RESET, WHITE
from tools.github_readme_sync.hierarchy import INDENTATION_UNIT
from tools.github_readme_sync.md import process_markdown
from tools.github_readme_sync.readme import ReadMe


def upload(new_hierarchy, file_path: str, rdme: ReadMe):
    logging.info(f"Uploading export folder: {file_path}")
    logging.info(f"URL: https://thousandbrainsproject.readme.io/v{rdme.version}/docs")
    rdme.create_version_if_not_exists()
    to_be_deleted = get_all_categories_docs(rdme)

    for category in new_hierarchy:
        cat_id, created = rdme.create_category_if_not_exists(
            category["slug"], category["title"]
        )
        logging.info(
            f"\n{BLUE}{category['title'].upper()}{GRAY}{created * ' [created]'}{RESET}"
        )

        set_do_not_delete(to_be_deleted, category["slug"])

        # Recursively process the hierarchy of children
        process_children(
            parent=category,
            cat_id=cat_id,
            file_path=file_path,
            rdme=rdme,
            to_be_deleted=to_be_deleted,
        )

    logging.info("")
    rdme.make_version_stable()

    if len(to_be_deleted) > 0:
        # Delete all docs and categories in reverse order
        for doc in reversed(to_be_deleted):
            if doc["type"] == "doc":
                rdme.delete_doc(doc["slug"])
            elif doc["type"] == "category":
                rdme.delete_category(doc["slug"])


def process_children(
    parent,
    cat_id,
    file_path,
    rdme,
    to_be_deleted,
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
        print_child(path_prefix.count("/"), doc, created)
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


def set_do_not_delete(to_be_deleted: list, slug: str):
    for doc in to_be_deleted:
        if doc["slug"] == slug:
            # remove the item from the list
            to_be_deleted.remove(doc)


def get_all_categories_docs(rdme: ReadMe):
    categories = rdme.get_categories()
    all_categories_and_docs = []
    for category in categories:
        all_categories_and_docs.append({"slug": category["slug"], "type": "category"})
        docs = rdme.get_category_docs(category)
        for doc in docs:
            all_categories_and_docs.append({"slug": doc["slug"], "type": "doc"})
            for child in doc["children"]:
                all_categories_and_docs.append({"slug": child["slug"], "type": "doc"})
                for sub_child in child["children"]:
                    all_categories_and_docs.append(
                        {"slug": sub_child["slug"], "type": "doc"}
                    )
    return all_categories_and_docs


def print_child(level: int, doc: dict, created: bool):
    color = CYAN if level else BLUE
    indent = INDENTATION_UNIT * level
    suffix = f"{GRAY}[created]{RESET}" if created else f"{GRAY}[updated]{RESET}"
    logging.info(
        f"{color}{indent}{doc['title']} {WHITE}/{doc['slug']} {GRAY}{suffix}{RESET}"
    )


def load_doc(file_path: str, category_slug: str, child: dict):
    file_path = os.path.join(file_path, category_slug, f"{child['slug']}.md")
    if not os.path.exists(file_path):
        raise ValueError(f"File {file_path} does not exist")

    with open(file_path, encoding="utf-8") as file:
        body = file.read()
        doc = process_markdown(body, child["slug"])
        return doc
