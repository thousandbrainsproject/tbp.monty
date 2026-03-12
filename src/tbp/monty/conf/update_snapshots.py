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
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from tbp.monty.frameworks.run_env import setup_env
from tbp.monty.hydra import register_resolvers

PROJECT_ROOT = Path(__file__).parents[4]


def update_snapshots(
    config_dir: Path,
    config_name: str = "experiment",
    override_key: str = "experiment",
    override_prefix: str = "",
    snapshots_dir: Path = PROJECT_ROOT / "tests" / "conf" / "snapshots",
):
    """Update snapshots for all configs in a directory.

    Args:
        config_dir: The directory containing the config YAML files.
        config_name: The Hydra config name (e.g. "experiment" or "test").
        override_key: The override key used in hydra.compose
            (e.g. "experiment" or "test").
        override_prefix: Prefix for the override value
            (e.g. "tutorial/" or "evidence_lm/").
        snapshots_dir: The directory to write the snapshots to.
    """
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    # Delete existing snapshots to remove renamed or deleted experiments
    for existing_snapshot in snapshots_dir.glob("*.yaml"):
        existing_snapshot.unlink()

    for file_path in config_dir.glob("*.yaml"):
        print(f"Updating snapshot: {file_path}")
        with hydra.initialize(version_base=None, config_path="."):
            config = hydra.compose(
                config_name=config_name,
                overrides=[f"{override_key}={override_prefix}{file_path.stem}"],
            )
            OmegaConf.to_object(config)
            current_config_yaml = OmegaConf.to_yaml(config)
            snapshot_path = snapshots_dir / f"{file_path.stem}.yaml"
            with snapshot_path.open("w") as f:
                f.write(current_config_yaml)


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    setup_env()
    register_resolvers()

    conf_dir = Path(__file__).parent
    snapshots_root = PROJECT_ROOT / "tests" / "conf" / "snapshots"

    # Experiment configs
    update_snapshots(
        config_dir=conf_dir / "experiment",
        snapshots_dir=snapshots_root,
    )
    update_snapshots(
        config_dir=conf_dir / "experiment" / "tutorial",
        override_prefix="tutorial/",
        snapshots_dir=snapshots_root / "tutorial",
    )

    # Below is commented out because we already generated the snapshots for
    # the tests, so we don't need to update them going forward. They are used
    # for comparison during the migration to new format and will be removed
    # after that is complete.

    # Test configs
    # test_subdirs = [
    #     "base_config",
    #     "evidence_lm",
    #     "frameworks/models/evidence_matching",
    #     "graph_building",
    #     "graph_learning",
    #     "hierarchy",
    #     "integration/positioning_procedures/get_good_view",
    #     "no_reset_evidence_lm",
    #     "policy",
    #     "profile",
    #     "sensor_module",
    # ]
    # for subdir in test_subdirs:
    #     update_snapshots(
    #         config_dir=conf_dir / "test" / subdir,
    #         config_name="test",
    #         override_key="test",
    #         override_prefix=f"{subdir}/",
    #         snapshots_dir=snapshots_root / "test" / subdir,
    #     )
