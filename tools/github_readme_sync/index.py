# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import json
import logging
import os
from pathlib import Path
from typing import Dict, List

from slugify import slugify

from tools.github_readme_sync.colors import CYAN, GREEN, RESET, YELLOW
from tools.github_readme_sync.file import find_markdown_files, read_file_content
from tools.github_readme_sync.md import parse_frontmatter, process_markdown

logger = logging.getLogger(__name__)


def _is_empty(value: str) -> bool:
    return not value or not value.strip()


def generate_index(docs_dir: str, output_file_path: str) -> str:
    """Generate index.json file from docs directory.

    Args:
        docs_dir: The directory containing markdown files to scan.
        output_file_path: Path where to write the output file.

    Returns:
        Path to the generated output file.

    Raises:
        ValueError: If docs_dir or output_file_path is empty.
    """
    if _is_empty(docs_dir):
        raise ValueError("docs_dir cannot be empty")
    if _is_empty(output_file_path):
        raise ValueError("output_file_path cannot be empty")

    logger.info(f"Scanning docs directory: {CYAN}{docs_dir}{RESET}")

    entries = process_markdown_files(docs_dir)

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    logger.info(
        f"{GREEN}Generated index with {len(entries)} entries: {output_file_path}{RESET}"
    )
    return output_file_path


def process_markdown_files(docs_dir: str) -> List[Dict]:
    """Process all markdown files in docs directory and extract front-matter.

    Args:
        docs_dir: The directory containing markdown files to scan.

    Continues if there are errors in the markdown files.

    Returns:
        List of dictionaries containing extracted front-matter and the body text.

    Raises:
        ValueError: If directory doesn't exist.
    """
    if _is_empty(docs_dir):
        raise ValueError("docs_dir cannot be empty")

    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise ValueError(f"Directory {docs_dir} does not exist")

    entries = []
    folder_name = docs_path.name

    for md_file in find_markdown_files(docs_dir):
        logger.info(f"Processing: {CYAN}{md_file.relative_to(docs_path)}{RESET}")

        try:
            content = read_file_content(md_file)
            frontmatter = parse_frontmatter(content)
        except (OSError, UnicodeDecodeError):
            logger.exception("Error reading %s", md_file)
            continue

        if not frontmatter:
            logger.warning(f"{YELLOW}No front-matter found in {md_file}{RESET}")
            continue

        processed_doc = process_markdown(content, slugify(md_file.stem))
        body_content = processed_doc["body"]

        relative_path = md_file.relative_to(docs_path)
        entry = {
            "title": frontmatter.get("title", ""),
            "slug": slugify(md_file.stem),
            "path": f"{folder_name}/{relative_path}",
            "text": body_content.strip(),
        }

        entry.update(
            {
                field: value
                for field, value in frontmatter.items()
                if field != "title" and value is not None
            }
        )

        entry.update(generate_path_components(md_file, docs_path))
        entries.append(entry)

    return entries


def generate_path_components(file_path: Path, docs_root: Path) -> Dict[str, str]:
    """Generate path components for a file relative to docs root.

    Returns:
        Dictionary with path1, path2, etc. keys for directory components.
    """
    relative_path = file_path.relative_to(docs_root)
    parts = relative_path.parts[:-1]

    path_components = {}
    for i, part in enumerate(parts):
        path_components[f"path{i + 1}"] = part

    return path_components
