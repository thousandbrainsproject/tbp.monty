# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from tbp.monty.frameworks.actions.actions import (
    LookDown,
    LookUp,
    MoveForward,
    MoveTangentially,
    OrientHorizontal,
    OrientVertical,
    SetAgentPitch,
    SetAgentPose,
    SetSensorPitch,
    SetSensorPose,
    SetSensorRotation,
    SetYaw,
    TurnLeft,
    TurnRight,
)
from tbp.monty.frameworks.actions.actuator import Actuator


class FakeActuator(Actuator):
    """A fake actuator that does nothing.

    Used for testing generic functionality that uses and interfaces with Actuators
    without considering specific actuator implementation details.
    """

    def actuate_look_down(self, _: LookDown) -> None:
        pass

    def actuate_look_up(self, _: LookUp) -> None:
        pass

    def actuate_move_forward(self, _: MoveForward) -> None:
        pass

    def actuate_move_tangentially(self, _: MoveTangentially) -> None:
        pass

    def actuate_orient_horizontal(self, _: OrientHorizontal) -> None:
        pass

    def actuate_orient_vertical(self, _: OrientVertical) -> None:
        pass

    def actuate_set_agent_pitch(self, _: SetAgentPitch) -> None:
        pass

    def actuate_set_agent_pose(self, _: SetAgentPose) -> None:
        pass

    def actuate_set_sensor_pitch(self, _: SetSensorPitch) -> None:
        pass

    def actuate_set_sensor_pose(self, _: SetSensorPose) -> None:
        pass

    def actuate_set_sensor_rotation(self, _: SetSensorRotation) -> None:
        pass

    def actuate_set_yaw(self, _: SetYaw) -> None:
        pass

    def actuate_turn_left(self, _: TurnLeft) -> None:
        pass

    def actuate_turn_right(self, _: TurnRight) -> None:
        pass
