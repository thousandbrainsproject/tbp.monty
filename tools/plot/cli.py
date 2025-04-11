# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import argparse
import logging
import sys
from pathlib import Path

monty_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(monty_root))

from tools.plot import objects_evidence_over_time, pose_error_over_time  # noqa: E402


def main():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
        dest="debug",
    )

    parser = argparse.ArgumentParser(
        description="Plot a figure", parents=[parent_parser]
    )
    subparsers = parser.add_subparsers(dest="command")

    objects_evidence_over_time.add_subparser(subparsers, parent_parser)
    pose_error_over_time.add_subparser(subparsers, parent_parser)

    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    logging.basicConfig(
        level=log_level,
        handlers=[handler],
    )
    logger = logging.getLogger(__name__)
    if args.debug:
        logger.debug("Debug logging enabled")

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
