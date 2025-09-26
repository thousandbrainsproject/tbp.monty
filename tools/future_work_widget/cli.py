# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .build import build

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    parser = argparse.ArgumentParser(
        description="Build the data and package the future work widget."
    )
    parser.add_argument("index_file", help="The JSON file to validate and transform")
    parser.add_argument(
        "output_dir", help="The output directory to create and save data.json"
    )
    parser.add_argument(
        "--docs-snippets-dir",
        help="Optional path to docs/snippets directory for validation files",
        default="docs/snippets",
    )

    args = parser.parse_args()

    index_file = Path(args.index_file)
    output_dir = Path(args.output_dir)
    docs_snippets_dir = Path(args.docs_snippets_dir)

    _validate_docs_snippets_dir(docs_snippets_dir)

    result = build(index_file, output_dir, docs_snippets_dir)

    _log_result(result)
    sys.exit(0 if result["success"] else 1)


def _validate_docs_snippets_dir(docs_snippets_dir: Path) -> None:
    if not docs_snippets_dir.exists():
        result = {
            "success": False,
            "error_message": f"Docs snippets directory not found: {docs_snippets_dir}",
        }
        _log_result(result)
        sys.exit(1)


def _log_result(result: dict) -> None:
    logger.info(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
