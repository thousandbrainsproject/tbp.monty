# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from typing import Any

import pytest
import yaml

from tests import HYDRA_ROOT

pytest.importorskip(
    "habitat_sim",
    reason="Habitat Sim optional dependency not installed.",
)

from pathlib import Path

import hydra
from omegaconf import OmegaConf
from unittest_parametrize import ParametrizedTestCase, param, parametrize

EXPERIMENT_DIR = (
    Path(__file__).parents[2] / "src" / "tbp" / "monty" / "conf" / "experiment"
)
EXPERIMENTS = [x.stem for x in EXPERIMENT_DIR.glob("*.yaml")]
EXPERIMENT_SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"

TUTORIALS_DIR = EXPERIMENT_DIR / "tutorial"
TUTORIALS = [x.stem for x in TUTORIALS_DIR.glob("*.yaml")]
TUTORIAL_SNAPSHOTS_DIR = Path(__file__).parent / "snapshots" / "tutorial"


def _config_mismatches(
    snapshot: dict[str, Any] | list[Any] | Any,
    current: dict[str, Any] | list[Any] | Any,
    path: str = "",
) -> list[str]:
    """Compare two configs recursively, returning dotted-path mismatch descriptions.

    Walks both structures in tandem, collecting human-readable descriptions
    of every difference found. Dict comparison is key-order invariant;
    list comparison is order-sensitive.

    Args:
        snapshot: The snapshot (expected) configuration value.
        current: The current (actual) configuration value.
        path: The dotted path to the current position in the config tree,
            used for generating readable mismatch descriptions.

    Returns:
        A list of mismatch description strings. Empty if the configs match.
    """
    mismatches = []
    if type(snapshot) is not type(current):
        mismatches.append(
            f"{path}: type mismatch: snapshot={type(snapshot).__name__}({snapshot!r}) "
            f"vs current={type(current).__name__}({current!r})"
        )
        return mismatches
    if isinstance(snapshot, dict):
        snapshot_keys = set(snapshot)
        current_keys = set(current)
        for k in snapshot_keys - current_keys:
            mismatches.append(f"{path}.{k}: missing in current config")
        for k in current_keys - snapshot_keys:
            mismatches.append(f"{path}.{k}: missing in snapshot")
        for k in snapshot_keys & current_keys:
            mismatches.extend(
                _config_mismatches(snapshot[k], current[k], f"{path}.{k}")
            )
        return mismatches
    if isinstance(snapshot, list):
        if len(snapshot) != len(current):
            mismatches.append(
                f"{path}: list length mismatch: "
                f"snapshot={len(snapshot)} vs current={len(current)}"
            )
            return mismatches
        for i, (snap_item, curr_item) in enumerate(zip(snapshot, current)):
            mismatches.extend(_config_mismatches(snap_item, curr_item, f"{path}[{i}]"))
        return mismatches
    if snapshot != current:
        mismatches.append(f"{path}: snapshot={snapshot!r} vs current={current!r}")
    return mismatches


def _assert_config_matches_snapshot(
    current_config_yaml: str, snapshot_config_yaml: str, name: str
):
    snapshot = yaml.safe_load(snapshot_config_yaml)
    current = yaml.safe_load(current_config_yaml)
    if snapshot != current:
        details = "\n".join(_config_mismatches(snapshot, current))
        raise AssertionError(
            f"\nThe {name} configuration does not match the stored snapshot.\n"
            f"Mismatches:\n{details}\n"
            "For more information on how to update snapshots"
            ", please see the tests/conf/README.md file."
        )


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
                snapshot_config_yaml = None
            if snapshot_config_yaml is not None:
                _assert_config_matches_snapshot(
                    current_config_yaml, snapshot_config_yaml, tutorial
                )
            else:
                with snapshot_path.open("w") as f:
                    f.write(current_config_yaml)
