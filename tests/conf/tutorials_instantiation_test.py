# Copyright 2026 Thousand Brains Project
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
from unittest_parametrize import ParametrizedTestCase, param, parametrize

EXPERIMENT_DIR = Path(__file__).parent.parent.parent / "conf" / "experiment"
TUTORIALS_DIR = EXPERIMENT_DIR / "tutorial"
TUTORIALS = [
    t
    for t in [x.stem for x in TUTORIALS_DIR.glob("*.yaml")]
    # skip omniglot tutorials due to pretrained model data dependencies
    if t not in ["omniglot_inference", "omniglot_training"]
]


class TutorialsTest(ParametrizedTestCase):
    @parametrize(
        "tutorial",
        [param(t, id=t) for t in TUTORIALS],
    )
    def test_tutorial(self, tutorial: str):
        with hydra.initialize(version_base=None, config_path="../../conf"):
            config = hydra.compose(
                config_name="experiment", overrides=[f"experiment=tutorial/{tutorial}"]
            )
            experiment = hydra.utils.instantiate(config.experiment)
            with experiment:
                pass
