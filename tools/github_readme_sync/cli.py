# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import argparse
import logging
import os
import sys
from os.path import dirname
from pathlib import Path

from dotenv import load_dotenv

monty_root = dirname(dirname(dirname(Path(__file__).resolve())))
sys.path.append(monty_root)

from tools.github_readme_sync.colors import RED, RESET  # noqa: E402
from tools.github_readme_sync.export import export  # noqa: E402
from tools.github_readme_sync.hierarchy import (  # noqa: E402
    check_external,
    check_hierarchy_file,
    create_hierarchy_file,
)
from tools.github_readme_sync.readme import ReadMe  # noqa: E402
from tools.github_readme_sync.upload import upload  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="CLI tool to manage exporting, checking, and uploading docs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Export command
    export_parser = subparsers.add_parser(
        "export", help="Export the readme docs and create a hierarchy.md file"
    )
    export_parser.add_argument(
        "folder", help="The directory where the exported docs will be stored"
    )
    export_parser.add_argument("version", help="The version for the exported docs")

    # Check command
    check_parser = subparsers.add_parser(
        "check", help="Check the hierarchy.md file and ensure all docs exist"
    )
    check_parser.add_argument(
        "folder", help="The directory containing hierarchy.md and corresponding docs"
    )

    # Upload command
    upload_parser = subparsers.add_parser(
        "upload",
        help="Upload the docs in the folder to ReadMe under the specified version",
    )
    upload_parser.add_argument("folder", help="The directory containing docs to upload")
    upload_parser.add_argument("version", help="The version to upload the docs under")

    # Check external links command
    check_external_parser = subparsers.add_parser(
        "check-external",
        help="Check external links in all markdown files from the specified directory",
    )
    check_external_parser.add_argument(
        "folder", help="The directory containing markdown files to check"
    )
    check_external_parser.add_argument(
        "version", help="The version to check external links for"
    )
    check_external_parser.add_argument(
        "--ignore",
        nargs="*",
        default=[],
        help="List of directories to exclude from link checking",
    )

    # Delete version command
    delete_parser = subparsers.add_parser(
        "delete", help="Delete a specific version from ReadMe"
    )
    delete_parser.add_argument("version", help="The version to delete")

    args = parser.parse_args()

    initialize()

    if args.command == "export":
        check_env()
        hierarchy = export(args.folder, ReadMe(args.version))
        create_hierarchy_file(args.folder, hierarchy)

    elif args.command == "check":
        check_hierarchy_file(args.folder)

    elif args.command == "upload":
        check_env()
        hierarchy = check_hierarchy_file(args.folder)
        upload(hierarchy, args.folder, rdme=ReadMe(args.version))

    elif args.command == "check-external":
        check_readme_api_key()
        check_external(args.folder, args.ignore, ReadMe(args.version))

    elif args.command == "delete":
        check_readme_api_key()
        rdme = ReadMe(args.version)
        rdme.delete_version()


def check_readme_api_key():
    if not os.getenv("README_API_KEY"):
        logging.error(f"{RED}README_API_KEY environment variable not set{RESET}")
        sys.exit(1)


def check_image_path():
    if not os.getenv("IMAGE_PATH"):
        logging.error(f"{RED}IMAGE_PATH environment variable not set{RESET}")
        sys.exit(1)


def check_env():
    check_readme_api_key()
    check_image_path()


def initialize():
    env_log_level = os.getenv("LOG_LEVEL")

    if env_log_level is None:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    else:
        logging.basicConfig(level=env_log_level.upper(), format="%(message)s")

    load_dotenv()


if __name__ == "__main__":
    main()
