# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import json
import logging
from pathlib import Path

import nh3
from slugify import slugify

from tools.github_readme_sync.colors import CYAN, GREEN, RESET, YELLOW
from tools.github_readme_sync.file import find_markdown_files, read_file_content
from tools.github_readme_sync.md import parse_frontmatter, process_markdown

logger = logging.getLogger(__name__)


def _is_empty(value: str) -> bool:
    return not value or not value.strip()


def _check_and_sanitize(
    key: str,
    value: str,
    max_key_length: int = 100,
    max_value_length: int = 10000,
) -> tuple[str, str] | None:
    """Sanitize and validate a key-value pair for length and injection attacks.

    Args:
        key: The key to sanitize and validate
        value: The value to sanitize and validate
        max_key_length: Maximum allowed length for keys
        max_value_length: Maximum allowed length for values

    Returns:
        Tuple of (sanitized_key, sanitized_value) if valid, None if invalid
    """
    if not isinstance(key, str) or not isinstance(value, str):
        return None

    if len(key) > max_key_length:
        logger.warning(
            f"Key '{key[:50]}...' exceeds maximum length of "
            f"{max_key_length} characters (actual: {len(key)})"
        )
        return None

    if len(value) > max_value_length:
        logger.warning(
            f"Value for key '{key}' exceeds maximum length of "
            f"{max_value_length} characters (actual: {len(value)})"
        )
        return None

    sanitized_key = nh3.clean(key).strip()
    sanitized_value = nh3.clean(str(value)).strip()

    if not sanitized_key or not sanitized_value:
        return None

    if sanitized_key != key:
        logger.info(f"Key sanitized: '{key}' -> '{sanitized_key}'")

    if sanitized_value != value:
        logger.info(
            f"Value for key '{key}' sanitized: '{value[:100]}...' -> "
            f"'{sanitized_value[:100]}...'"
        )

    return sanitized_key, sanitized_value


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

    Path(output_file_path).parent.mkdir(exist_ok=True, parents=True)
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    logger.info(
        f"{GREEN}Generated index with {len(entries)} entries: {output_file_path}{RESET}"
    )
    return output_file_path


def process_markdown_files(docs_dir: str) -> list[dict]:
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

    docs_path = Path(docs_dir).resolve()
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

        path_components = generate_path_components(md_file, docs_path)
        sources_to_sanitize = [
            (frontmatter, lambda k, v: k != "title" and v is not None),
            (path_components, lambda _k, _v: True),
        ]

        for source_dict, filter_func in sources_to_sanitize:
            for key, value in source_dict.items():
                if filter_func(key, value):
                    sanitized_entry = _check_and_sanitize(key, str(value))
                    if sanitized_entry:
                        sanitized_key, sanitized_value = sanitized_entry
                        entry[sanitized_key] = sanitized_value
        entries.append(entry)

    return entries


def generate_path_components(file_path: Path, docs_root: Path) -> dict[str, str]:
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
