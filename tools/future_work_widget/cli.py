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
import logging
import sys
from pathlib import Path

monty_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(monty_root))

# Note: this tool requires this specific import order so don't remove
# the noqa: E402 comments
from tools.future_work_widget.build import build  # noqa: E402

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
        help="Optional path to a snippets directory for validation files",
        default=Path("docs/snippets"),
    )

    args = parser.parse_args()

    index_file = Path(args.index_file)
    output_dir = Path(args.output_dir)
    docs_snippets_dir = Path(args.docs_snippets_dir)

    result = build(index_file, output_dir, docs_snippets_dir)

    logger.info(result.model_dump_json(exclude_none=True, indent=2))
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
