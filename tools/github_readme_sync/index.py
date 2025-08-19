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
import re
from pathlib import Path
from typing import Dict, List

from slugify import slugify

from tools.github_readme_sync.colors import CYAN, GREEN, RED, RESET, YELLOW
from tools.github_readme_sync.file_utils import find_markdown_files, read_file_content
from tools.github_readme_sync.md import parse_frontmatter


class FrontMatterValidator:
    """Validates front-matter fields according to RFC requirements."""

    @classmethod
    def _matches_pattern(cls, value: str, patterns: List[str]) -> bool:
        """Check if value matches any regex pattern or exact string.

        Returns:
            True if value matches any pattern, False otherwise.
        """
        return any(
            re.match(p, value) if p.startswith("^") else value == p for p in patterns
        )

    @classmethod
    def validate(cls, frontmatter: Dict) -> List[str]:
        """Validate front-matter and return list of errors.

        Returns:
            List of validation error messages.
        """
        errors = []

        validations = {
            "status": (
                [r"^completed$", r"^in-progress$", r"^none$"],
                "completed, in-progress, none",
            ),
            "size": (
                [r"^small$", r"^medium$", r"^large$", r"^unknown$"],
                "small, medium, large, unknown",
            ),
            "rfc": (
                [
                    r"^required$",
                    r"^optional$",
                    r"^not-required$",
                    r"^(?:https://)?github\.com/thousandbrainsproject/tbp\.monty/pull/\d+$",
                ],
                "required, optional, not-required or a GitHub pull request link",
            ),
        }

        for field, (patterns, options) in validations.items():
            if field in frontmatter and not cls._matches_pattern(
                frontmatter[field], patterns
            ):
                errors.append(
                    f"Invalid {field} '{frontmatter[field]}'. Must be one of: {options}"
                )

        for field, max_count in [("tags", 10), ("skills", 10)]:
            if field in frontmatter:
                items = parse_comma_separated_field(frontmatter[field])
                if len(items) > max_count:
                    errors.append(
                        f"Too many {field} ({len(items)}). Maximum allowed: {max_count}"
                    )

        return errors


def parse_comma_separated_field(value) -> List[str]:
    """Parse a field that might be comma-separated or already a list.

    Args:
        value: Field value (string, list, or other)

    Returns:
        List of cleaned string values
    """
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    items = str(value).split(",") if isinstance(value, str) else [str(value)]
    return [item.strip() for item in items if item.strip()]


def generate_path_components(file_path: Path, docs_root: Path) -> Dict[str, str]:
    """Generate path components for a file relative to docs root.

    Returns:
        Dictionary with path1, path2, etc. keys for directory components.
    """
    relative_path = file_path.relative_to(docs_root)
    parts = relative_path.parts[:-1]

    path_components = {}
    for i, part in enumerate(parts):
        path_components[f"path{i+1}"] = part

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
    errors_found = False
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

        validation_errors = FrontMatterValidator.validate(frontmatter)
        if validation_errors:
            errors_found = True
            rel_path = md_file.relative_to(docs_path)
            logging.error(f"{RED}Validation errors in {rel_path}:{RESET}")
            for error in validation_errors:
                logging.error(f"  - {error}")
            continue

        relative_path = md_file.relative_to(docs_path)
        entry = {
            "title": frontmatter.get("title", ""),
            "slug": slugify(md_file.stem),
            "path": f"{folder_name}/{relative_path}",
        }

        simple_fields = ["group", "size", "rfc", "status", "rfc-link"]
        for field in simple_fields:
            if field in frontmatter and frontmatter[field] is not None:
                entry[field] = frontmatter[field]

        for field in ["tags", "skills", "implementation"]:
            if field in frontmatter:
                parsed = parse_comma_separated_field(frontmatter[field])
                if parsed:
                    entry[field] = parsed

        entry.update(generate_path_components(md_file, docs_path))
        entries.append(entry)

    if errors_found:
        raise ValueError("Validation errors found. Please fix the front-matter issues.")

    return entries


def generate_index(docs_dir: str) -> str:
    """Generate index.json file from docs directory.

    Returns:
        Path to the generated index.json file.
    """
    output_file = os.path.join(docs_dir, "index.json")
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
