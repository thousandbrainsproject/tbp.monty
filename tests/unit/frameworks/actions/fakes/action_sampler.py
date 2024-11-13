# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from tbp.monty.frameworks.actions.action_samplers import ActionSampler
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


class FakeSampler(ActionSampler):
    def sample_look_down(self, agent_id: str) -> LookDown:
        pass

    def sample_look_up(self, agent_id: str) -> LookUp:
        pass

    def sample_move_forward(self, agent_id: str) -> MoveForward:
        pass

    def sample_move_tangentially(self, agent_id: str) -> MoveTangentially:
        pass

    def sample_orient_horizontal(self, agent_id: str) -> OrientHorizontal:
        pass

    def sample_orient_vertical(self, agent_id: str) -> OrientVertical:
        pass

    def sample_set_agent_pitch(self, agent_id: str) -> SetAgentPitch:
        pass

    def sample_set_agent_pose(self, agent_id: str) -> SetAgentPose:
        pass

    def sample_set_sensor_pitch(self, agent_id: str) -> SetSensorPitch:
        pass

    def sample_set_sensor_pose(self, agent_id: str) -> SetSensorPose:
        pass

    def sample_set_sensor_rotation(self, agent_id: str) -> SetSensorRotation:
        pass

    def sample_set_yaw(self, agent_id: str) -> SetYaw:
        pass

    def sample_turn_left(self, agent_id: str) -> TurnLeft:
        pass

    def sample_turn_right(self, agent_id: str) -> TurnRight:
        pass
