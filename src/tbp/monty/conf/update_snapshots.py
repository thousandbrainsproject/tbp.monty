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

import sys
from argparse import ArgumentParser
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from tbp.monty.frameworks.run_env import setup_env
from tbp.monty.hydra import register_resolvers

PROJECT_ROOT = Path(__file__).parents[4]


def update_snapshots(
    config_dir: Path,
    config_name: str = "experiment",
    override_prefix: str = "",
    snapshots_dir: Path = PROJECT_ROOT / "tests" / "conf" / "snapshots",
    generate_mujoco: bool = False,
):
    """Update snapshots for all configs in a directory.

    Args:
        config_dir: The directory containing the config YAML files.
        config_name: The Hydra config name (e.g. "experiment").
        override_prefix: Prefix for the override value
            (e.g. "tutorial/" or "evidence_lm/").
        snapshots_dir: The directory to write the snapshots to.
        generate_mujoco: Whether to we're generating MuJoCo or Habitat snapshots.
    """
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    snapshot_names = {s.name for s in snapshots_dir.glob("*.yaml")}

    for file_path in config_dir.glob("*.yaml"):
        snapshot_names.discard(file_path.name)

        # TODO: remove once we remove Habitat
        is_mujoco = file_path.stem.endswith("mujoco")
        if is_mujoco != generate_mujoco:
            # This is either a MuJoCo config and we're not generating MuJoCo configs
            # or a Habitat config and we are generating MuJoCo configs, so skip it.
            continue

        print(f"Updating snapshot: {file_path}")
        with hydra.initialize(version_base=None, config_path="."):
            print(f"experiment={override_prefix}{file_path.stem}")
            config = hydra.compose(
                config_name=config_name,
                overrides=[f"experiment={override_prefix}{file_path.stem}"],
            )
            OmegaConf.to_object(config)
            current_config_yaml = OmegaConf.to_yaml(config)
            snapshot_path = snapshots_dir / f"{file_path.stem}.yaml"
            with snapshot_path.open("w") as f:
                f.write(current_config_yaml)

    # Delete remaining snapshots to remove renamed or deleted experiments
    for name in snapshot_names:
        print(f"Removing snapshot: {name}")
        path = snapshots_dir / name
        path.unlink()


if __name__ == "__main__":
    arg_parser = ArgumentParser(
        prog="update_snapshots.py",
        description="Updates config snapshots used by conf_test.py",
    )
    arg_parser.add_argument(
        "-m",
        "--mujoco",
        action="store_true",
        default=False,
        help="Generate MuJoCo configs instead of Habitat configs",
    )
    args = arg_parser.parse_args()

    sys.path.insert(0, str(PROJECT_ROOT))
    setup_env()
    register_resolvers()

    conf_dir = Path(__file__).parent
    snapshots_root = PROJECT_ROOT / "tests" / "conf" / "snapshots"

    # Experiment configs
    update_snapshots(
        config_dir=conf_dir / "experiment",
        snapshots_dir=snapshots_root,
        generate_mujoco=args.mujoco,
    )
    update_snapshots(
        config_dir=conf_dir / "experiment" / "tutorial",
        override_prefix="tutorial/",
        snapshots_dir=snapshots_root / "tutorial",
        generate_mujoco=args.mujoco,
    )
