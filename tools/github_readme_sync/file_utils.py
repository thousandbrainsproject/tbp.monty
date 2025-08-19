# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os
from typing import List, Optional

DEFAULT_IGNORE_DIRS = [".pytest_cache", ".github", ".git", "figures", "snippets"]
DEFAULT_IGNORE_FILES = ["hierarchy.md"]


def find_markdown_files(
    folder: str,
    ignore_dirs: Optional[List[str]] = None,
    ignore_files: Optional[List[str]] = None
) -> List[str]:
    """Find all markdown files in a directory, excluding specified dirs and files.

    Args:
        folder: Root directory to search
        ignore_dirs: List of directory names to exclude (uses defaults if None)
        ignore_files: List of file names to exclude (uses defaults if None)

    Returns:
        List of full paths to markdown files
    """
    if ignore_dirs is None:
        ignore_dirs = DEFAULT_IGNORE_DIRS.copy()
    if ignore_files is None:
        ignore_files = DEFAULT_IGNORE_FILES.copy()

    md_files = []
    for root, _, files in os.walk(folder):
        # Skip directories that should be ignored
        if any(ignore_dir in root for ignore_dir in ignore_dirs):
            continue

        # Find markdown files, excluding ignored files
        for file in files:
            if file.endswith(".md") and file not in ignore_files:
                md_files.append(os.path.join(root, file))

    return md_files


def read_file_content(file_path: str) -> str:
    """Read file content with UTF-8 encoding.

    Args:
        file_path: Path to the file to read

    Returns:
        File content as string
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
