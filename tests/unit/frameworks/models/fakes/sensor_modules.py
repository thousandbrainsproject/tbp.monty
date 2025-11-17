# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import numpy as np

from tbp.monty.frameworks.models.abstract_monty_classes import SensorModule
from tbp.monty.frameworks.models.states import State


class FakeSensorModule(SensorModule):
    """Dummy placeholder class used only for tests."""

    def __init__(
        self,
        rng,  # noqa: ARG002
        sensor_module_id: str,
    ):
        super().__init__()
        self.sensor_module_id = sensor_module_id

    def state_dict(self):
        pass

    def update_state(self, state):
        pass

    def pre_episode(self):
        pass

    def post_episode(self):
        pass

    def set_experiment_mode(self, mode: str):
        pass

    def step(self, data):
        """Returns a dummy/placeholder state."""
        return State(
            location=np.zeros(3),
            morphological_features={
                "pose_vectors": np.eye(3),
                "pose_fully_defined": True,
            },
            non_morphological_features=data,
            confidence=1,
            use_state=True,
            sender_id=self.sensor_module_id,
            sender_type="SM",
        )
