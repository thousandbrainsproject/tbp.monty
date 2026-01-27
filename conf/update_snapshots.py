# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Updates test snapshots from the current config.

Usage:
    python update_snapshots.py
"""

from pathlib import Path

import hydra
from omegaconf import OmegaConf

from tbp.monty.frameworks.run_env import setup_env
from tbp.monty.hydra import register_resolvers


def update_snapshots(
    experiment_dir: Path = Path(__file__).parent / "experiment",
    experiment_prefix: str = "",
    snapshots_dir: Path = Path(__file__).parent.parent / "tests" / "conf" / "snapshots",
):
    """Update snapshots for all experiments in the experiment directory.

    Args:
        experiment_dir: The directory containing the experiments.
        experiment_prefix: The prefix to add to the experiment name (e.g. "tutorial/")
        snapshots_dir: The directory to write the snapshots to.
    """
    # Delete existing snapshots to remove renamed or deleted experiments
    for existing_snapshot in snapshots_dir.glob("*.yaml"):
        existing_snapshot.unlink()

    for file_path in experiment_dir.glob("*.yaml"):
        print(f"Updating snapshot: {file_path}")
        with hydra.initialize(version_base=None, config_path="."):
            config = hydra.compose(
                config_name="experiment",
                overrides=[f"experiment={experiment_prefix}{file_path.stem}"],
            )
            OmegaConf.to_object(config)
            current_config_yaml = OmegaConf.to_yaml(config)
            snapshot_path = snapshots_dir / f"{file_path.stem}.yaml"
            with snapshot_path.open("w") as f:
                f.write(current_config_yaml)


if __name__ == "__main__":
    setup_env()
    register_resolvers()
    update_snapshots()
    update_snapshots(
        experiment_dir=Path(__file__).parent / "experiment" / "tutorial",
        experiment_prefix="tutorial/",
        snapshots_dir=(
            Path(__file__).parent.parent / "tests" / "conf" / "snapshots" / "tutorial"
        ),
    )
