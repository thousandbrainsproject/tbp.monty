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
from pathlib import Path

from .build import build

monty_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(monty_root))


def main():
    parser = argparse.ArgumentParser(
        description="CLI tool to manage future work widget."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build
    build_parser = subparsers.add_parser(
        "build", help="Build the data and package the widget"
    )
    build_parser.add_argument(
        "index_file", help="The index.json file to process"
    )
    build_parser.add_argument(
        "output_dir", help="The output directory to create and save data.json"
    )
    build_parser.add_argument(
        "--docs-snippets-dir",
        help="Path to docs/snippets directory for validation files",
        default="docs/snippets",
    )

    args = parser.parse_args()

    initialize()

    if args.command == "build":
        docs_snippets_dir = args.docs_snippets_dir
        try:
            build(args.index_file, args.output_dir, docs_snippets_dir)
        except ValueError as e:
            # Handle validation errors gracefully without stack trace
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            # For unexpected errors, show the full traceback
            logging.exception("Unexpected error during build")
            sys.exit(1)


def initialize():
    env_log_level = os.getenv("LOG_LEVEL")

    if env_log_level is None:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    else:
        logging.basicConfig(level=env_log_level.upper(), format="%(message)s")





if __name__ == "__main__":
    main()
