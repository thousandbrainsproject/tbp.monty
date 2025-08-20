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

    args = parser.parse_args()

    initialize()

    if args.command == "build":
        build(args.index_file, args.output_dir)


def initialize():
    env_log_level = os.getenv("LOG_LEVEL")

    if env_log_level is None:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    else:
        logging.basicConfig(level=env_log_level.upper(), format="%(message)s")





if __name__ == "__main__":
    main()
