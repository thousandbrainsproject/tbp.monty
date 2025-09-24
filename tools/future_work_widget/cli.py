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
import os
import sys
from pathlib import Path

from .build import build

sys.path.append(str(Path(__file__).resolve().parent.parent))
from github_readme_sync.colors import RED, RESET

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
    build_parser.add_argument("index_file", help="The index.json file to process")
    build_parser.add_argument(
        "output_dir", help="The output directory to create and save data.json"
    )
    build_parser.add_argument(
        "--docs-snippets-dir",
        help="Path to docs/snippets directory for validation files",
        default="docs/snippets",
    )
    build_parser.add_argument(
        "--json",
        action="store_true",
        help="Output validation results in JSON format for CI/CD integration",
    )

    args = parser.parse_args()

    initialize()

    if args.command == "build":
        docs_snippets_dir = args.docs_snippets_dir

        snippets_path = Path(docs_snippets_dir)
        if not snippets_path.exists():
            error_msg = f"Docs snippets directory not found: {docs_snippets_dir}"
            if args.json:
                result = {
                    "success": False,
                    "processed_items": 0,
                    "total_items": 0,
                    "errors": [
                        {
                            "message": error_msg,
                            "file": "cli",
                            "line": 1,
                            "field": None,
                            "level": "error",
                            "title": "DirectoryNotFoundError",
                            "annotation_level": "failure",
                        }
                    ],
                    "error_message": error_msg,
                }
                print(json.dumps(result, indent=2))
            else:
                print(f"{RED}Error: {error_msg}{RESET}", file=sys.stderr)
            sys.exit(1)

        if args.json:
            logging.getLogger().setLevel(logging.CRITICAL)

        result = build(args.index_file, args.output_dir, docs_snippets_dir)

        if args.json:
            print(json.dumps(result, indent=2))
        elif not result["success"]:
            error_count = len(result["errors"])
            print(
                f"{RED}Error: Validation failed with {error_count} error(s):{RESET}",
                file=sys.stderr,
            )
            print(file=sys.stderr)

            for i, error in enumerate(result["errors"], 1):
                file_path = error["file"]
                line = error["line"]
                message = error["message"]

                print(f"{file_path}:{line}", file=sys.stderr)
                print(f"{message}", file=sys.stderr)
                if i < error_count:
                    print(file=sys.stderr)

        sys.exit(0 if result["success"] else 1)


def initialize():
    env_log_level = os.getenv("LOG_LEVEL")

    if env_log_level is None:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    else:
        logging.basicConfig(level=env_log_level.upper(), format="%(message)s")


if __name__ == "__main__":
    main()
