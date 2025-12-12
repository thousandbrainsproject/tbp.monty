# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import pytest

pytest.importorskip(
    "habitat_sim",
    reason="Habitat Sim optional dependency not installed.",
)

from pathlib import Path

import hydra
from omegaconf import OmegaConf
from unittest_parametrize import ParametrizedTestCase, param, parametrize

EXPERIMENT_DIR = Path(__file__).parent.parent.parent / "conf" / "experiment"
EXPERIMENTS = [x.stem for x in EXPERIMENT_DIR.glob("*.yaml")]
EXPERIMENT_SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"

TUTORIALS_DIR = EXPERIMENT_DIR / "tutorial"
TUTORIALS = [x.stem for x in TUTORIALS_DIR.glob("*.yaml")]
TUTORIAL_SNAPSHOTS_DIR = Path(__file__).parent / "snapshots" / "tutorial"


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
        with hydra.initialize(version_base=None, config_path="../../conf"):
            config = hydra.compose(
                config_name="experiment", overrides=[f"experiment={experiment}"]
            )
            # force resolving the config for any parsing errors
            OmegaConf.to_object(config)
            current_config_yaml = OmegaConf.to_yaml(config)
            try:
                snapshot_config_yaml = snapshot_path.read_text()
            except FileNotFoundError:
                snapshot_config_yaml = None
            if snapshot_config_yaml is not None:
                _assert_config_matches_snapshot(
                    current_config_yaml, snapshot_config_yaml, experiment
                )
            else:
                with snapshot_path.open("w") as f:
                    f.write(current_config_yaml)


class TutorialTest(ParametrizedTestCase):
    @parametrize(
        "tutorial",
        [param(t, id=t) for t in TUTORIALS],
    )
    def test_tutorial(self, tutorial: str):
        snapshot_path = TUTORIAL_SNAPSHOTS_DIR / f"{tutorial}.yaml"
        with hydra.initialize(version_base=None, config_path="../../conf"):
            config = hydra.compose(
                config_name="experiment", overrides=[f"experiment=tutorial/{tutorial}"]
            )
            # force resolving the config for any parsing errors
            OmegaConf.to_object(config)
            current_config_yaml = OmegaConf.to_yaml(config)
            try:
                snapshot_config_yaml = snapshot_path.read_text()
            except FileNotFoundError:
                snapshot_config_yaml = None
            if snapshot_config_yaml is not None:
                _assert_config_matches_snapshot(
                    current_config_yaml, snapshot_config_yaml, tutorial
                )
            else:
                with snapshot_path.open("w") as f:
                    f.write(current_config_yaml)
