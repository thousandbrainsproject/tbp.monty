# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_IGNORE_DIRS = ["figures", "snippets"]
DEFAULT_IGNORE_FILES = ["hierarchy.md"]


def get_folders(file_path: str) -> list[str]:
    """Get a list of folder names in the specified directory.

    The order of the returned folder names is not guaranteed.

    Args:
        file_path: Path to the directory to list folders from

    Returns:
        List of folder names in the directory
    """
    return [child.name for child in Path(file_path).iterdir() if child.is_dir()]


def find_markdown_files(
    folder: str,
    ignore_dirs: list[str] | None = None,
    ignore_files: list[str] | None = None,
) -> list[Path]:
    """Find all markdown files in a directory, excluding specified dirs and files.

    Args:
        folder: Root directory to search
        ignore_dirs: List of directory names to exclude (uses defaults if None)
        ignore_files: List of file names to exclude (uses defaults if None)

    Returns:
        List of Path objects to markdown files
    """
    ignore_dirs = DEFAULT_IGNORE_DIRS if ignore_dirs is None else ignore_dirs
    ignore_files = DEFAULT_IGNORE_FILES if ignore_files is None else ignore_files

    md_files = []
    for root, _, files in os.walk(folder):
        path_parts = Path(root).parts
        if any(part.startswith(".") for part in path_parts):
            continue
        if any(ignore_dir in path_parts for ignore_dir in ignore_dirs):
            continue

        md_files.extend(
            Path(root).joinpath(file)
            for file in files
            if file.endswith(".md") and file not in ignore_files
        )
    return md_files


def read_file_content(file_path: str) -> str:
    """Read file content with UTF-8 encoding.

    Args:
        file_path: Path to the file to read

    Returns:
        File content as string
    """
    with open(file_path, encoding="utf-8") as f:
        return f.read()
