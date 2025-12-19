# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Update a single snapshot from benchmarks_v2 experiments.

Usage:
    python conf/update_snapshot_benchmarks_v2.py base_config_10distinctobj_dist_agent
    python conf/update_snapshot_benchmarks_v2.py base_config_10distinctobj_dist_agent.yaml
"""

import argparse
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from tbp.monty.frameworks.run_env import setup_env
from tbp.monty.hydra import register_resolvers


def update_single_snapshot(
    experiment_name: str,
    experiment_dir: Path,
    experiment_prefix: str,
    snapshots_dir: Path,
):
    """Update snapshot for a single experiment.

    Args:
        experiment_name: The name of the experiment (without .yaml extension).
        experiment_dir: The directory containing the experiments.
        experiment_prefix: The prefix to add to the experiment name (e.g. "benchmarks_v2/").
        snapshots_dir: The directory to write the snapshot to.
    """
    # Remove .yaml extension if present
    if experiment_name.endswith(".yaml"):
        experiment_name = experiment_name[:-5]

    experiment_path = experiment_dir / f"{experiment_name}.yaml"
    if not experiment_path.exists():
        raise FileNotFoundError(
            f"Experiment file not found: {experiment_path}"
        )

    print(f"Updating snapshot: {experiment_path}")
    # Since we're in conf/, config_path="." works like in update_snapshots.py
    with hydra.initialize(version_base=None, config_path="."):
        config = hydra.compose(
            config_name="experiment",
            overrides=[f"experiment={experiment_prefix}{experiment_name}"],
        )
        OmegaConf.to_object(config)
        current_config_yaml = OmegaConf.to_yaml(config)
        snapshot_path = snapshots_dir / f"{experiment_name}.yaml"
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        with snapshot_path.open("w") as f:
            f.write(current_config_yaml)
        print(f"Snapshot saved to: {snapshot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Update a snapshot from benchmarks_v2 experiments"
    )
    parser.add_argument(
        "experiment",
        type=str,
        help="Name of the experiment file (with or without .yaml extension)",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path(__file__).parent / "experiment" / "benchmarks_v2",
        help="Directory containing experiment files (default: conf/experiment/benchmarks_v2)",
    )
    parser.add_argument(
        "--snapshots-dir",
        type=Path,
        default=Path(__file__).parent.parent / "tests" / "conf" / "snapshots_v2",
        help="Directory to write snapshots to (default: tests/conf/snapshots_v2)",
    )
    parser.add_argument(
        "--experiment-prefix",
        type=str,
        default="benchmarks_v2/",
        help="Prefix to add to experiment name (default: benchmarks_v2/)",
    )

    args = parser.parse_args()

    setup_env()
    register_resolvers()

    update_single_snapshot(
        experiment_name=args.experiment,
        experiment_dir=args.experiment_dir,
        experiment_prefix=args.experiment_prefix,
        snapshots_dir=args.snapshots_dir,
    )


if __name__ == "__main__":
    main()
