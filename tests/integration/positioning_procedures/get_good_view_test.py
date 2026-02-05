# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import tempfile
import unittest

import hydra
import numpy as np
from omegaconf import DictConfig

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.environments.environment import SemanticID
from tbp.monty.frameworks.environments.positioning_procedures import (
    get_perc_on_obj_semantic,
)
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.experiments.object_recognition_experiments import (
    MontyObjectRecognitionExperiment,
)
from tests import HYDRA_ROOT


def hydra_config(test_name: str, output_dir: str) -> DictConfig:
    return hydra.compose(
        config_name="test",
        overrides=[
            f"test=integration/positioning_procedures/get_good_view/{test_name}",
            f"test.config.logging.output_dir={output_dir}",
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
            exp: MontyObjectRecognitionExperiment = hydra.utils.instantiate(config.test)
            with exp:
                exp.experiment_mode = ExperimentMode.TRAIN
                exp.model.set_experiment_mode(exp.experiment_mode)
                exp.pre_epoch()
                exp.pre_episode()

                policy_cfg = config.test.config["monty_config"]["motor_system_config"][
                    "motor_system_args"
                ]["policy"]
                target_perc_on_target_obj = policy_cfg["good_view_percentage"]
                target_closest_point = policy_cfg["desired_object_distance"]

                ctx = RuntimeContext(rng=exp.rng)
                observation = exp.env_interface.step(ctx, first=True)
                view = observation[exp.model.motor_system._policy.agent_id][
                    "view_finder"
                ]
                semantic = view["semantic_3d"][:, 3].reshape(view["depth"].shape)
                perc_on_target_obj = get_perc_on_obj_semantic(
                    semantic, semantic_id=SemanticID(1)
                )

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

    def test_multi_object_target_not_visible(self) -> None:
        """Test the ability to move a distant agent to a good view of an object.

        Given a view of multiple objects with the target object not visible, the
        positioning procedure should generate distant agent actions to
        a good view of the object before beginning the experiment.

        In this case, there are multiple objects, such that at the start of the
        experiment the target object is not visible in the central pixel of the view.
        Positioning procedure must turn the agent towards the target object using the
        viewfinder, and then move towards it.
        """
        with hydra.initialize_config_dir(version_base=None, config_dir=str(HYDRA_ROOT)):
            config = hydra_config("multi_object_target_not_visible", self.output_dir)
            exp: MontyObjectRecognitionExperiment = hydra.utils.instantiate(config.test)
            with exp:
                exp.train()

                exp.experiment_mode = ExperimentMode.EVAL
                exp.model.set_experiment_mode(exp.experiment_mode)
                exp.pre_epoch()
                exp.pre_episode()

                policy_cfg = config.test.config["monty_config"]["motor_system_config"][
                    "motor_system_args"
                ]["policy"]
                target_perc_on_target_obj = policy_cfg["good_view_percentage"]
                target_closest_point = policy_cfg["desired_object_distance"]

                ctx = RuntimeContext(rng=exp.rng)
                observation = exp.env_interface.step(ctx, first=True)
                view = observation[exp.model.motor_system._policy.agent_id][
                    "view_finder"
                ]
                semantic = view["semantic_3d"][:, 3].reshape(view["depth"].shape)
                perc_on_target_obj = get_perc_on_obj_semantic(
                    semantic, semantic_id=SemanticID(1)
                )

                assert perc_on_target_obj >= target_perc_on_target_obj, (
                    f"Initial view is not good enough, {perc_on_target_obj} "
                    f"vs target of {target_perc_on_target_obj}"
                )

                points_on_target_obj = semantic == 1
                closest_point_on_target_obj = np.min(
                    view["depth"][points_on_target_obj]
                )

                assert closest_point_on_target_obj > target_closest_point, (
                    "Initial view is too close to target, "
                    f"{closest_point_on_target_obj} vs target of {target_closest_point}"
                )
                # Also calculate the closest point on *any* object so that we don't get
                # too close and clip into objects; An object is anything with
                # SemanticID > 0
                points_on_any_obj = view["semantic"] > 0
                closest_point_on_any_obj = np.min(view["depth"][points_on_any_obj])
                assert closest_point_on_any_obj > target_closest_point / 6, (
                    "Initial view too close to other objects, "
                    f"{closest_point_on_any_obj} vs target of "
                    f"{target_closest_point / 6}"
                )
