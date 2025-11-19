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
import shutil

from slugify import slugify

from tools.github_readme_sync.colors import BLUE, CYAN, RESET
from tools.github_readme_sync.hierarchy import INDENTATION_UNIT
from tools.github_readme_sync.readme import ReadMe


def export(output_dir: str, rdme: ReadMe):
    hierarchy = []
    categories = rdme.get_categories()

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    for i, category in enumerate(categories):
        category_entry = {
            "title": category["title"],
            "slug": slugify(category["title"]),
            "children": [],
        }
        hierarchy.append(category_entry)

        logging.info(
            "\n" if i > 0 else "" + f"{BLUE}{slugify(category['title']).upper()}{RESET}"
        )

        category_folder_path = os.path.join(output_dir, slugify(category["title"]))
        os.makedirs(category_folder_path, exist_ok=True)

        docs_from_server = rdme.get_category_docs(category)
        for server_doc in docs_from_server:
            hierarchy_doc = {
                "title": server_doc["title"],
                "slug": slugify(server_doc["title"]),
                "children": [],
            }
            category_entry["children"].append(hierarchy_doc)

            # Call process_doc with named parameters
            process_doc(
                server_doc=server_doc,
                hierarchy_doc=hierarchy_doc,
                folder_path=category_folder_path,
                indent_level=0,
                rdme=rdme,
            )

    return hierarchy


def process_doc(*, server_doc, hierarchy_doc, folder_path, indent_level, rdme):
    indent = INDENTATION_UNIT * indent_level
    logging.info(f"{indent}{CYAN}{hierarchy_doc['slug']}{RESET}")

    doc_path = os.path.join(folder_path, f"{hierarchy_doc['slug']}.md")
    with open(doc_path, "w") as f:
        f.write(rdme.get_doc_by_slug(server_doc["slug"]))

    children = server_doc.get("children", [])
    if children:
        child_folder_path = os.path.join(folder_path, hierarchy_doc["slug"])
        os.makedirs(child_folder_path, exist_ok=True)

    for child in children:
        child_entry = {
            "title": child["title"],
            "slug": slugify(child["title"]),
            "children": [],
        }
        hierarchy_doc["children"].append(child_entry)

        # Process the child document recursively with increased indent level
        process_doc(
            server_doc=child,
            hierarchy_doc=child_entry,
            folder_path=child_folder_path,
            indent_level=indent_level + 1,
            rdme=rdme,
        )
