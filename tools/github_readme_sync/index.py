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
from typing import Dict, List, Optional

from slugify import slugify

from tools.github_readme_sync.colors import CYAN, GREEN, RED, RESET, YELLOW
from tools.github_readme_sync.file_utils import find_markdown_files, read_file_content
from tools.github_readme_sync.md import parse_frontmatter


class FrontMatterValidator:
    """Validates front-matter fields according to RFC requirements."""

    VALID_STATUS = ["completed", "in-progress", "none"]
    VALID_SIZE = ["small", "medium", "large", "unknown"]
    VALID_RFC = ["required", "optional", "not-required"]
    MAX_TAGS = 10
    MAX_SKILLS = 10

    @classmethod
    def validate(cls, frontmatter: Dict) -> List[str]:
        """Validate front-matter and return list of errors.

        Returns:
            List of validation error messages.
        """
        errors = []

        # Validate status field
        if "status" in frontmatter:
            if frontmatter["status"] not in cls.VALID_STATUS:
                errors.append(
                    f"Invalid status '{frontmatter['status']}'. "
                    f"Must be one of: {', '.join(cls.VALID_STATUS)}"
                )
        # Validate size field
        if "size" in frontmatter:
            if frontmatter["size"] not in cls.VALID_SIZE:
                errors.append(
                    f"Invalid size '{frontmatter['size']}'. "
                    f"Must be one of: {', '.join(cls.VALID_SIZE)}"
                )

        # Validate rfc field
        if "rfc" in frontmatter:
            if frontmatter["rfc"] not in cls.VALID_RFC:
                errors.append(
                    f"Invalid rfc '{frontmatter['rfc']}'. "
                    f"Must be one of: {', '.join(cls.VALID_RFC)}"
                )

        # Validate tags limit
        if "tags" in frontmatter:
            tags = frontmatter["tags"]
            if isinstance(tags, list) and len(tags) > cls.MAX_TAGS:
                errors.append(
                    f"Too many tags ({len(tags)}). "
                    f"Maximum allowed: {cls.MAX_TAGS}"
                )

        # Validate skills limit
        if "skills" in frontmatter:
            skills = frontmatter["skills"]
            if isinstance(skills, list) and len(skills) > cls.MAX_SKILLS:
                errors.append(
                    f"Too many skills ({len(skills)}). "
                    f"Maximum allowed: {cls.MAX_SKILLS}"
                )

        return errors


def extract_frontmatter_from_file(file_path: str) -> Optional[Dict]:
    """Extract front-matter from a markdown file.

    Returns:
        Dictionary containing front-matter or None if error occurred.
    """
    try:
        content = read_file_content(file_path)
        return parse_frontmatter(content)
    except (OSError, UnicodeDecodeError):
        logging.exception(f"Error reading {file_path}")
        return None


def generate_path_components(file_path: Path, docs_root: Path) -> Dict[str, str]:
    """Generate path components for a file relative to docs root.

    Returns:
        Dictionary with path1, path2, etc. keys for directory components.
    """
    relative_path = file_path.relative_to(docs_root)
    parts = relative_path.parts[:-1]  # Exclude the file name itself

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

    # Find all markdown files using shared utility
    md_files = find_markdown_files(docs_dir)

    for md_file_path in md_files:
        md_file = Path(md_file_path)
        docs_path = Path(docs_dir)

        logging.info(f"Processing: {CYAN}{md_file.relative_to(docs_path)}{RESET}")

        # Extract front-matter
        frontmatter = extract_frontmatter_from_file(md_file_path)
        if frontmatter is None:
            logging.warning(f"{YELLOW}No front-matter found in {md_file}{RESET}")
            continue

        # Validate front-matter
        validation_errors = FrontMatterValidator.validate(frontmatter)

        if validation_errors:
            errors_found = True
            logging.error(
                f"{RED}Validation errors in "
                f"{md_file.relative_to(docs_path)}:{RESET}"
            )
            for error in validation_errors:
                logging.error(f"  - {error}")
            continue

        # Generate entry
        entry = {
            "title": frontmatter.get("title", ""),
            "slug": slugify(md_file.stem),
        }

        # Add optional fields if present
        optional_fields = [
            "group", "tags", "size", "rfc", "status",
            "rfc-link", "implementation", "skills"
        ]
        for field in optional_fields:
            if field in frontmatter:
                entry[field] = frontmatter[field]

        # Generate path components
        path_components = generate_path_components(md_file, docs_path)
        entry.update(path_components)

        entries.append(entry)

    if errors_found:
        raise ValueError(
            "Validation errors found. Please fix the front-matter issues."
        )

    return entries


def generate_index(docs_dir: str) -> str:
    """Generate index.json file from docs directory.

    Returns:
        Path to the generated index.json file.
    """
    output_file = os.path.join(docs_dir, "index.json")

    logging.info(f"Scanning docs directory: {CYAN}{docs_dir}{RESET}")

    # Process all markdown files
    entries = process_markdown_files(docs_dir)

    # Sort entries by path1, then path2, then title
    entries.sort(key=lambda x: (
        x.get("path1", ""),
        x.get("path2", ""),
        x.get("title", "")
    ))

    # Write to JSON file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    logging.info(
        f"{GREEN}Generated index with {len(entries)} entries: "
        f"{output_file}{RESET}"
    )
    return output_file
