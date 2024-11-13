# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import List, Optional

import attr
import habitat_sim.utils as hab_utils
import magnum as mn
from habitat_sim.agent.controls.controls import ActuationSpec, SceneNodeControl
from habitat_sim.agent.controls.default_controls import _move_along, _rotate_local
from habitat_sim.registry import registry
from habitat_sim.scene import SceneNode

__all__ = [
    "SetYaw",
    "SetSensorPitch",
    "SetAgentPitch",
    "SetAgentPose",
    "SetSensorRotation",
    "SetSensorPose",
]


_X_AXIS = 0
_Y_AXIS = 1
_Z_AXIS = 2


@attr.s(auto_attribs=True)  # TODO check significance of this decorator
class ActuationVecSpec(ActuationSpec):
    """Inherits from Meta's habitat-sim class.

    Expects a list of lists.

    Enables passing in two lists to set_pose --> the first is the xyz absolute positions
    for the agent, and the second list contains the coefficients of a quaternion
    specifying absolute rotation
    """

    amount: List[List[float]]
    constraint: Optional[float] = None


def _move_along_diagonal(
    scene_node: SceneNode, distance: float, direction: list
) -> None:
    ax = mn.Vector3(direction)
    scene_node.translate_local(ax * distance)


@registry.register_move_fn(body_action=True)
class SetYaw(SceneNodeControl):
    """Custom habitat-sim action used to set the agent body absolute yaw rotation.

    :class:`ActuationSpec` amount contains the new absolute yaw rotation in degrees.
    """

    def __call__(self, scene_node: SceneNode, actuation_spec: ActuationSpec) -> None:
        angle = mn.Deg(actuation_spec.amount)

        # Since z+ is out of the page, x+ is "right", and y+ is "up", then changes
        # in yaw should presumably be about the y-axis, otherwise we might change
        # the roll. TODO investigate this further and update the original
        # implementation if necessary.
        new_rotation = mn.Quaternion.rotation(angle, mn.Vector3.z_axis())
        scene_node.rotation = new_rotation.normalized()


@registry.register_move_fn(body_action=False)
class SetSensorPitch(SceneNodeControl):
    """Custom habitat-sim action used to set the *sensor* absolute pitch rotation.

    Note this does not update the pitch of the agent (imagine e.g. the "body"
    associated with the eye remaining in place, but the eye moving).

    :class:`ActuationSpec` amount contains the new absolute pitch rotation in degrees.
    """

    def __call__(self, scene_node: SceneNode, actuation_spec: ActuationSpec) -> None:
        angle = mn.Deg(actuation_spec.amount)

        # Since z+ is out of the page, x+ is "right", and y+ is "up", then changes
        # in pitch should presumably be about the x-axis (similar to LookUp and
        # LookDown in default_controls.py, but in absolute rather than relative
        # coordinates). TODO investigate this further and update the original
        # implementation if necessary.
        new_rotation = mn.Quaternion.rotation(angle, mn.Vector3.y_axis())
        scene_node.rotation = new_rotation.normalized()


@registry.register_move_fn(body_action=True)
class SetAgentPitch(SceneNodeControl):
    """Custom habitat-sim action used to set the *agent* absolute pitch rotation.

    Note that unless otherwise changed, the sensor maintains identity orientation w.r.t
    the agent, so in effect, this will also adjust the pitch of the sensor w.r.t the
    environment.

    This difference in behavior is controlled by the body_action boolean set in the
    decorator.

    :class:`ActuationSpec` amount contains the new absolute pitch rotation in degrees.

    TODO add unit test to habitat_sim_test.py
    """

    def __call__(self, scene_node: SceneNode, actuation_spec: ActuationSpec) -> None:
        angle = mn.Deg(actuation_spec.amount)

        # As for SetPitchSenosr, likely need to change the axis about which to rotate;
        # TODO it would also be worth investigating the significance of rotating the
        # sensor vs. agent, and establish more clearly which of these we would prefer;
        # in the original set-pitch and set-yaw implementations, pitch affects the
        # sensor, while yaw affects the agent
        new_rotation = mn.Quaternion.rotation(angle, mn.Vector3.y_axis())

        scene_node.rotation = new_rotation.normalized()


@registry.register_move_fn(body_action=False)
class SetSensorPose(SceneNodeControl):
    """Custom habitat-sim action used to set the sensor pose.

    This action sets the sensor's absolute location (xyz coordinate), and rotation
    relative to the agent.

    Note that body_action=False, resulting in a sensor (rather than agent) update.

    :class:`ActuationSpec` amount contains location and rotation (latter as a numpy
        quaternion)
    """

    def __call__(self, scene_node: SceneNode, actuation_spec: ActuationVecSpec) -> None:
        # Note setting scene_node.translation applies an absolute location, not a
        # relative translation (here in agent-centric coordinates)
        scene_node.translation = mn.Vector3(
            [
                actuation_spec.amount[0][0],
                actuation_spec.amount[0][1],
                actuation_spec.amount[0][2],
            ]
        )

        magnum_quat = hab_utils.common.quat_to_magnum(actuation_spec.amount[1])
        scene_node.rotation = magnum_quat


@registry.register_move_fn(body_action=False)
class SetSensorRotation(SceneNodeControl):
    """Custom habitat-sim action used to set the sensor rotation.

    This action sets the sensor's absolute rotation relative to the agent.

    Note body_action=False, resulting in a sensor (rather than agent) update

    :class:`ActuationSpec` amount contains rotation (as a numpy quaternion)
    """

    def __call__(self, scene_node: SceneNode, actuation_spec: ActuationVecSpec) -> None:
        magnum_quat = hab_utils.common.quat_to_magnum(actuation_spec.amount[0])
        scene_node.rotation = magnum_quat


@registry.register_move_fn(body_action=True)
class SetAgentPose(SceneNodeControl):
    """Custom habitat-sim action used to set the agent pose.

    This action sets the agent body absolute location (xyz coordinate), and absolute
    rotation (i.e. relative to the identity rotation, which is defined by the axes of
    xyz in Habitat) in the environment.

    :class:`ActuationSpec` amount contains location and rotation (latter as a numpy
        quaternion)
    """

    def __call__(self, scene_node: SceneNode, actuation_spec: ActuationVecSpec) -> None:
        # Note setting scene_node.translation applies an absolute location, not a
        # relative translation (here in global environmental coordinates)
        scene_node.translation = mn.Vector3(
            [
                actuation_spec.amount[0][0],
                actuation_spec.amount[0][1],
                actuation_spec.amount[0][2],
            ]
        )

        magnum_quat = hab_utils.common.quat_to_magnum(actuation_spec.amount[1])
        scene_node.rotation = magnum_quat


@registry.register_move_fn(body_action=True)
class MoveForward(SceneNodeControl):
    def __call__(self, scene_node: SceneNode, actuation_spec: ActuationSpec) -> None:
        _move_along(scene_node, -actuation_spec.amount, _Z_AXIS)


@registry.register_move_fn(body_action=True)
class MoveTangentially(SceneNodeControl):
    def __call__(self, scene_node: SceneNode, actuation_spec: ActuationSpec) -> None:
        _move_along_diagonal(
            scene_node, actuation_spec.amount, direction=actuation_spec.constraint
        )


@registry.register_move_fn(body_action=True)
class OrientHorizontal(SceneNodeControl):
    def __call__(self, scene_node: SceneNode, actuation_spec: ActuationSpec) -> None:
        # Use the constraint parameter to move the agent left, compensating for the
        # right turn such that the same center point is fixated upon
        _move_along(scene_node, -actuation_spec.constraint[0], _X_AXIS)
        _rotate_local(scene_node, -actuation_spec.amount, _Y_AXIS, constraint=None)
        _move_along(scene_node, -actuation_spec.constraint[1], _Z_AXIS)


@registry.register_move_fn(body_action=True)
class OrientVertical(SceneNodeControl):
    def __call__(self, scene_node: SceneNode, actuation_spec: ActuationSpec) -> None:
        # Use the constraint parameter to move the agent down and forward,
        # compensating for the upward turn such that the same center point is fixated
        # upon
        _move_along(scene_node, -actuation_spec.constraint[0], _Y_AXIS)
        _rotate_local(scene_node, actuation_spec.amount, _X_AXIS, constraint=None)
        _move_along(scene_node, -actuation_spec.constraint[1], _Z_AXIS)
