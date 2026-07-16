# Copyright 2025-2026 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import logging
import shutil
from pathlib import Path

from slugify import slugify

from tools.github_readme_sync.colors import BLUE, CYAN, RESET
from tools.github_readme_sync.hierarchy import INDENTATION_UNIT
from tools.github_readme_sync.readme import ReadMe

logger = logging.getLogger(__name__)


def export(output_dir: str, rdme: ReadMe):
    output_dir = Path(output_dir)
    hierarchy = []
    categories = rdme.get_categories()

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)

    for i, category in enumerate(categories):
        # API v2 categories no longer have server-side slugs.
        # Generate a local slug from the category title for the folder
        # name and hierarchy.md.
        category_slug = slugify(category["title"])

        category_entry = {
            "title": category["title"],
            "slug": category_slug,
            "children": [],
        }
        hierarchy.append(category_entry)

        prefix = "\n" if i > 0 else ""
        logger.info(f"{prefix}{BLUE}{category_slug.upper()}{RESET}")

        category_folder_path = output_dir / category_slug
        category_folder_path.mkdir(exist_ok=True, parents=True)

        # API v2 returns category pages as a flat collection.
        # Rebuild the hierarchy using each page's parent URI.
        docs_from_server = rdme.get_category_doc_tree(category)

        for server_doc in docs_from_server:
            hierarchy_doc = {
                "title": server_doc["title"],
                # Preserve the actual ReadMe slug. Do not derive a slug
                # from the page title.
                "slug": server_doc["slug"],
                "children": [],
            }
            category_entry["children"].append(hierarchy_doc)

            process_doc(
                server_doc=server_doc,
                hierarchy_doc=hierarchy_doc,
                folder_path=category_folder_path,
                indent_level=0,
                rdme=rdme,
            )

    return hierarchy


def process_doc(
    *,
    server_doc,
    hierarchy_doc,
    folder_path,
    indent_level,
    rdme,
):
    indent = INDENTATION_UNIT * indent_level
    logger.info(f"{indent}{CYAN}{hierarchy_doc['slug']}{RESET}")

    doc_path = folder_path / f"{hierarchy_doc['slug']}.md"

    # Documentation may contain Unicode text.
    with doc_path.open("w", encoding="utf-8") as f:
        f.write(rdme.get_doc_by_slug(server_doc["slug"]))

    children = server_doc.get("children", [])

    if children:
        child_folder_path = folder_path / hierarchy_doc["slug"]
        child_folder_path.mkdir(exist_ok=True, parents=True)

    for child in children:
        child_entry = {
            "title": child["title"],
            # Preserve the child page's actual ReadMe slug.
            "slug": child["slug"],
            "children": [],
        }
        hierarchy_doc["children"].append(child_entry)

        process_doc(
            server_doc=child,
            hierarchy_doc=child_entry,
            folder_path=child_folder_path,
            indent_level=indent_level + 1,
            rdme=rdme,
        )
