# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from pathlib import Path

import hydra
import pytest
from omegaconf import OmegaConf
from unittest_parametrize import ParametrizedTestCase, param, parametrize

from tests import HYDRA_ROOT

EXPERIMENT_DIR = (
    Path(__file__).parents[3] / "src" / "tbp" / "monty" / "conf" / "experiment"
)
EXPERIMENTS = [
    x.stem
    for x in EXPERIMENT_DIR.glob("*.yaml")
    # Only check MuJoCo experiments
    # TODO: Merge with original once we convert to MuJoCo
    if x.stem.endswith("mujoco")
]
EXPERIMENT_SNAPSHOTS_DIR = Path(__file__).parents[2] / "conf" / "snapshots"

TUTORIALS_DIR = EXPERIMENT_DIR / "tutorial"
TUTORIALS = [
    x.stem
    for x in TUTORIALS_DIR.glob("*.yaml")
    # Only check MuJoCo tutorials
    # TODO: Merge with original once we convert to MuJoCo
    if x.stem.endswith("mujoco")
]
TUTORIAL_SNAPSHOTS_DIR = EXPERIMENT_SNAPSHOTS_DIR / "tutorial"


def _assert_config_matches_snapshot(
    current_config_yaml: str, snapshot_config_yaml: str, name: str
):
    try:
        assert snapshot_config_yaml == current_config_yaml
    except AssertionError as e:
        # TODO: Use e.add_note() once we get to Python 3.11 or higher
        raise AssertionError(
            f"\nThe {name} configuration does not match the stored "
            "snapshot.\nPlease see the direct cause exception above for details about "
            "the mismatch.\nFor more information on how to update snapshots"
            ", please see the tests/conf/README.md file."
        ) from e


class ExperimentTest(ParametrizedTestCase):
    @parametrize(
        "experiment",
        [param(e, id=e) for e in EXPERIMENTS],
    )
    def test_experiment(self, experiment: str):
        snapshot_path = EXPERIMENT_SNAPSHOTS_DIR / f"{experiment}.yaml"
        with hydra.initialize_config_dir(version_base=None, config_dir=str(HYDRA_ROOT)):
            config = hydra.compose(
                config_name="experiment", overrides=[f"experiment={experiment}"]
            )
            # force resolving the config for any parsing errors
            OmegaConf.to_object(config)
            current_config_yaml = OmegaConf.to_yaml(config)

            try:
                snapshot_config_yaml = snapshot_path.read_text()
            except FileNotFoundError:
                pytest.fail(f"Missing snapshot file for '{experiment}'")

            _assert_config_matches_snapshot(
                current_config_yaml, snapshot_config_yaml, experiment
            )


class TutorialTest(ParametrizedTestCase):
    @parametrize(
        "tutorial",
        [param(t, id=t) for t in TUTORIALS],
    )
    def test_tutorial(self, tutorial: str):
        snapshot_path = TUTORIAL_SNAPSHOTS_DIR / f"{tutorial}.yaml"
        with hydra.initialize_config_dir(version_base=None, config_dir=str(HYDRA_ROOT)):
            config = hydra.compose(
                config_name="experiment", overrides=[f"experiment=tutorial/{tutorial}"]
            )
            # force resolving the config for any parsing errors
            OmegaConf.to_object(config)
            current_config_yaml = OmegaConf.to_yaml(config)

            try:
                snapshot_config_yaml = snapshot_path.read_text()
            except FileNotFoundError:
                pytest.fail(f"Missing snapshot file for '{tutorial}'")
            _assert_config_matches_snapshot(
                current_config_yaml, snapshot_config_yaml, tutorial
            )
