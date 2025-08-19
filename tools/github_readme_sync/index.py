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
from tools.github_readme_sync.md import parse_frontmatter


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


def process_markdown_files(docs_dir: str) -> List[Dict]:
    """Process all markdown files in docs directory and extract front-matter.

    Returns:
        List of dictionaries containing extracted front-matter and metadata.

    Raises:
        ValueError: If docs directory doesn't exist or validation errors found.
    """
    if not os.path.exists(docs_dir):
        raise ValueError(f"Directory {docs_dir} does not exist")

    entries = []
    docs_path = Path(docs_dir)
    folder_name = os.path.basename(docs_dir)

    for md_file_path in find_markdown_files(docs_dir):
        md_file = Path(md_file_path)
        logging.info(f"Processing: {CYAN}{md_file.relative_to(docs_path)}{RESET}")

        try:
            content = read_file_content(md_file_path)
            frontmatter = parse_frontmatter(content)
        except (OSError, UnicodeDecodeError):
            logging.exception(f"Error reading {md_file_path}")
            continue

        if not frontmatter:
            logging.warning(f"{YELLOW}No front-matter found in {md_file}{RESET}")
            continue

        relative_path = md_file.relative_to(docs_path)
        entry = {
            "title": frontmatter.get("title", ""),
            "slug": slugify(md_file.stem),
            "path": f"{folder_name}/{relative_path}",
        }

        entry.update(
            {
                field: value
                for field, value in frontmatter.items()
                if field not in ["title"] and value is not None
            }
        )

        entry.update(generate_path_components(md_file, docs_path))
        entries.append(entry)

    return entries


def generate_index(docs_dir: str, output_file_path: str) -> str:
    """Generate index.json file from docs directory.

    Args:
        docs_dir: The directory containing markdown files to scan.
        output_file_path: Path where to write the index.json file.

    Returns:
        Path to the generated index.json file.
    """
    output_file = output_file_path
    logging.info(f"Scanning docs directory: {CYAN}{docs_dir}{RESET}")

    entries = process_markdown_files(docs_dir)
    entries.sort(
        key=lambda x: (x.get("path1", ""), x.get("path2", ""), x.get("title", ""))
    )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    logging.info(
        f"{GREEN}Generated index with {len(entries)} entries: {output_file}{RESET}"
    )
    return output_file
