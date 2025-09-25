# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import argparse
import json
import logging
import sys
from pathlib import Path

from .build import build

logger = logging.getLogger(__name__)


def _validate_docs_snippets_dir(docs_snippets_dir: str) -> None:
    snippets_path = Path(docs_snippets_dir)
    if not snippets_path.exists():
        error_msg = f"Docs snippets directory not found: {docs_snippets_dir}"
        result = {
            "success": False,
            "error_message": error_msg,
        }
        logger.info(json.dumps(result, indent=2))
        sys.exit(1)


def main():
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="CLI tool to manage future work widget."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build", help="Build the data and package the widget"
    )
    build_parser.add_argument("index_file", help="The index.json file to process")
    build_parser.add_argument(
        "output_dir", help="The output directory to create and save data.json"
    )
    build_parser.add_argument(
        "--docs-snippets-dir",
        help="Path to docs/snippets directory for validation files",
        default="docs/snippets",
    )

    args = parser.parse_args()

    if args.command == "build":
        _validate_docs_snippets_dir(args.docs_snippets_dir)

        result = build(
            Path(args.index_file), Path(args.output_dir), Path(args.docs_snippets_dir)
        )

        logger.info(json.dumps(result, indent=2))
        sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
