# Copyright 2025-2026 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Tuple

from habitat_sim.agent import ActionSpec, ActuationSpec

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.simulators.habitat import HabitatAgent

from .config import DIGIT, TactoSensorSpec
from .sensors import TactoSensor

__all__ = ["TactoFingerAgent"]

Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]
Size = Tuple[int, int]


class TactoFingerAgent(HabitatAgent):
    """A simple tacto sensor mounted to the tip of a single finger.

    With the following predefined actions:

        - "move_forward": Move finger forward using `translation_step`
        - "move_backward": Move finger backward using `translation_step`
        - "turn_left": Turn finger left using `rotation_step`
        - "turn_right": Turn finger right using `rotation_step`

    Attributes:
        agent_id: Unique ID used to identify this agent's actions and observations.
        position: Agent initial position in meters. Default (0, 0, 0).
        rotation: Agent initial rotation quaternion. Default (1, 0, 0, 0).
        height: Agent height in meters. Default 0.1 (10 cm).
        rotation_step: Rotation step in degrees for the "turn" actions.
        translation_step: Translation step in meters for the "move" actions.
        config: Tacto sensor configuration ('OMNITACT', 'DIGIT'). Default 'DIGIT'.
    """

    def __init__(
        self,
        agent_id: AgentID,
        sensor_id: SensorID,
        position: Vector3 = (0.0, 0.0, 0.0),
        rotation: Quaternion = (1.0, 0.0, 0.0, 0.0),
        height: float = 1.5,
        resolution: Size = (32, 48),
        rotation_step: float = 0.0,
        translation_step: float = 0.0,
        config: TactoSensorSpec = DIGIT,
    ):
        super().__init__(agent_id, position, rotation, height)

        self.sensor_id = sensor_id
        self.resolution = resolution
        self.rotation_step = rotation_step
        self.translation_step = translation_step

        self.sensors.append(
            TactoSensor(
                sensor_id=self.sensor_id,
                resolution=self.resolution,
                position=self.position,
                rotation=self.rotation,
                config=config,
            )
        )

    def get_spec(self):
        spec = super().get_spec()
        spec.action_space = {
            f"{self.agent_id}.move_forward": ActionSpec(
                "move_forward", ActuationSpec(amount=self.translation_step)
            ),
            f"{self.agent_id}.move_backward": ActionSpec(
                "move_forward", ActuationSpec(amount=-self.translation_step)
            ),
            f"{self.agent_id}.turn_left": ActionSpec(
                "turn_left", ActuationSpec(amount=self.rotation_step)
            ),
            f"{self.agent_id}.turn_right": ActionSpec(
                "turn_right", ActuationSpec(amount=self.rotation_step)
            ),
        }
        return spec
