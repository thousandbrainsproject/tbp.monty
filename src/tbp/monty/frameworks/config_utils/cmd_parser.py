# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import argparse


def create_cmd_parser(experiments: list[str]):
    """Create monty command line argument parser from all available configs.

    Args:
        experiments: List of experiment names available to choose from.

    Returns:
        Command line argument parser
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=True,
    )
    # Run options
    parser.add_argument(
        "-e",
        "--experiments",
        nargs="+",
        help="Experiment names",
        choices=experiments,
    )
    parser.add_argument(
        "-q",
        "--quiet_habitat_logs",
        default=True,
        help="Set logging levels in habitat to quiet",
    )
    parser.add_argument(
        "-p",
        "--print_config",
        action="store_true",
        help="Don't run an experiment; just print out the config for visual inspection",
    )

    return parser


def create_rerun_parser(experiments: list[str]):
    """Create command line argument parser for running.

    Args:
        experiments: List of experiment names available to choose from.

    Returns:
        Command line argument parser
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=True,
    )
    # Run options
    parser.add_argument(
        "-e",
        "-exp",
        "--experiment",
        help="Name of the experiment you want to repeat episodes from",
        choices=experiments,
    )
    parser.add_argument(
        "-i",
        "--idx",
        "--episodes",
        nargs="+",
        default=[],
        dest="episodes",
        help="List of int, indices of episodes you want to re-run. Index lookup will be"
        " based on file names from reproducibility logger, "
        "e.g. reproduce_episode_data/eval_episode_2_actions.jsonl",
    )
    parser.add_argument(
        "-n",
        "--name",
        default="rerun",
        dest="name",
        help="name of follow up config that will be appended to parent name",
    )

    return parser


def create_cmd_parser_parallel(experiments: list[str]):
    """Create monty command line argument parser for running episodes in parallel.

    This one is designed to run episodes of an experiment in parallel and is used
    by run_parallel.py.

    Args:
        experiments: List of experiment names available to choose from.

    Returns:
        Command line argument parser
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=True,
    )
    # Run options
    parser.add_argument(
        "-e",
        "--experiment",
        help="Experiment name",
        choices=experiments,
    )
    parser.add_argument(
        "-i",
        "--episodes",
        default="all",
        help=(
            "Which episodes to run (zero-based). Examples: "
            "'all' | '3' | '0,3,5:8' | '4:' | ':8'. "
        ),
        type=str,
    )
    parser.add_argument(
        "-n",
        "--num_parallel",
        default=16,
        help="How many episodes to run in parallel",
        type=int,
    )
    parser.add_argument(
        "-q",
        "--quiet_habitat_logs",
        default=True,
        help="Set logging levels in habitat to quiet",
    )
    parser.add_argument(
        "-p",
        "--print_cfg",
        action="store_true",
        help="Don't run an experiment; just print out the config for visual inspection",
    )

    return parser
