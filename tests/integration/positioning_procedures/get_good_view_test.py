# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import pytest

from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.hydra import instantiate_experiment

pytest.importorskip(
    "habitat_sim",
    reason="Habitat Sim optional dependency not installed.",
)

import tempfile
import unittest

import hydra
import numpy as np
from omegaconf import DictConfig

from tbp.monty.frameworks.environments.environment import SemanticID
from tbp.monty.frameworks.environments.positioning_procedures import (
    GOOD_VIEW_DISTANCE_DEFAULT,
    GOOD_VIEW_PERCENTAGE_DEFAULT,
    get_perc_on_obj_semantic,
)
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tests import HYDRA_ROOT


def hydra_config(test_name: str, output_dir: str) -> DictConfig:
    return hydra.compose(
        config_name="experiment",
        overrides=[
            f"experiment=test/integration/positioning_procedures/get_good_view/{test_name}",
            f"experiment.config.logging.output_dir={output_dir}",
        ],
    )


class GetGoodViewTest(unittest.TestCase):
    def setUp(self) -> None:
        self.output_dir = tempfile.mkdtemp()

    def test_dist_agent_too_far_away(self) -> None:
        """Test the ability to move a distant agent to a good view of an object.

        Given a too far away view of an object, the positioning procedure should
        generate distant agent actions to a good view of the object before beginning
        the experiment.

        In this case, the object is a bit too far away, and so the agent moves forward.
        """
        with hydra.initialize_config_dir(version_base=None, config_dir=str(HYDRA_ROOT)):
            config = hydra_config("dist_agent_too_far_away", self.output_dir)
            agent_cfg = config.experiment.config.environment.env_init_args.agents
            agent_id = config.experiment.config.train_env_interface_args[
                "positioning_procedures"
            ][0].agent_id
            exp = instantiate_experiment(config.experiment)
            with exp:
                exp.experiment_mode = ExperimentMode.TRAIN
                exp.model.set_experiment_mode(exp.experiment_mode)
                exp.pre_epoch()
                exp.pre_episode()

                target_perc_on_target_obj = GOOD_VIEW_PERCENTAGE_DEFAULT
                target_closest_point = GOOD_VIEW_DISTANCE_DEFAULT

                assert exp.env_interface is not None  # for type narrowing
                observation, state = exp.env_interface.step([])
                view = observation[agent_id][SensorID("view_finder")]
                semantic = view["semantic_3d"][:, 3].reshape(view["depth"].shape)
                perc_on_target_obj = get_perc_on_obj_semantic(
                    semantic, semantic_id=SemanticID(1)
                )

                # Make sure we've actually moved
                assert not np.allclose(
                    state[agent_id].position, agent_cfg.agent_args.position
                ), "Agent did not move."

                assert perc_on_target_obj >= target_perc_on_target_obj, (
                    f"Initial view is not good enough, {perc_on_target_obj} "
                    f"vs target of {target_perc_on_target_obj}"
                )
                points_on_target_obj = semantic == 1
                closest_point_on_target_obj = np.min(
                    view["depth"][points_on_target_obj]
                )

                assert closest_point_on_target_obj > target_closest_point, (
                    f"Initial view is too close, {closest_point_on_target_obj} "
                    f"vs target of {target_closest_point}"
                )
