# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import numpy as np
import quaternion as qt
import scipy.ndimage
from scipy.spatial.transform import Rotation as rot  # noqa: N813
from typing_extensions import Protocol

from tbp.monty.frameworks.actions.actions import Action, LookDown, MoveForward, TurnLeft
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environments.environment import SemanticID
from tbp.monty.frameworks.models.motor_system_state import MotorSystemState
from tbp.monty.frameworks.sensors import SensorID

__all__ = [
    "GetGoodView",
    "PositioningProcedure",
    "PositioningProcedureResult",
]

logger = logging.getLogger(__name__)


@dataclass
class PositioningProcedureResult:
    """Result of a positioning procedure.

    For more on the terminated/truncated terminology, see https://farama.org/Gymnasium-Terminated-Truncated-Step-API.
    """

    actions: list[Action] = field(default_factory=list)
    """Actions to take."""
    success: bool = False
    """Whether the procedure succeeded in its positioning goal."""
    terminated: bool = False
    """Whether the procedure reached a terminal state with success or failure."""
    truncated: bool = False
    """Whether the procedure was truncated due to a limit on the number of attempts or
    other criteria."""


class PositioningProcedure(Protocol):
    """Positioning procedure to position the agent in the scene.

    The positioning procedure should be called repeatedly until the procedure result
    indicates that the procedure has terminated or truncated.
    """

    @staticmethod
    def depth_at_center(agent_id: AgentID, observation: Any, sensor_id: str) -> float:
        """Determine the depth of the central pixel for the sensor.

        Args:
            agent_id: The ID of the agent to use.
            observation: The observation to use.
            sensor_id: The ID of the sensor to use.

        Returns:
            The depth of the central pixel for the sensor.
        """
        # TODO: A lot of assumptions are made here about the shape of the observation.
        #       This should be made robust.
        observation_shape = observation[agent_id][sensor_id]["depth"].shape
        return observation[agent_id][sensor_id]["depth"][
            observation_shape[0] // 2, observation_shape[1] // 2
        ]

    def __call__(
        self,
        observation: Mapping,
        state: MotorSystemState,
    ) -> PositioningProcedureResult:
        """Return a list of actions to position the agent in the scene.

        Args:
            observation: The observation to use for positioning.
            state: The current state of the motor system.

        Returns:
            Any actions to take, whether the procedure succeeded, whether the procedure
            terminated, and whether the procedure truncated.
        """
        pass


def get_perc_on_obj_semantic(
    semantic_obs, semantic_id: SemanticID | Literal["any"] = "any"
):
    """Get the percentage of pixels in the observation that land on the target object.

    If a semantic ID is provided, then only pixels on the target object are counted;
    otherwise, pixels on any object are counted.

    This uses the semantic image, where each pixel is associated with a semantic ID
    that is unique for each object, and always >0.

    Args:
        semantic_obs: Semantic image observation.
        semantic_id: Semantic ID of the target object. If "any", then pixels belonging
            to any object are counted. Defaults to "any".

    Returns:
        perc_on_obj: Percentage of pixels on the object.
    """
    res = semantic_obs.shape[0] * semantic_obs.shape[1]
    if semantic_id == "any":
        csum = np.sum(semantic_obs >= 1)
    else:
        # Count only pixels on the target (e.g. primary target) object
        csum = np.sum(semantic_obs == semantic_id)
    return csum / res


class GetGoodView(PositioningProcedure):
    """Positioning procedure to get a good view of the object before an episode.

    Used to position the distant agent so that it finds the initial view of an object
    at the beginning of an episode with respect to a given sensor (the surface agent
    is positioned using the TouchObject positioning procedure instead). Also currently
    used by the distant agent after a "jump" has been initialized by a model-based
    policy.

    First, the agent is moved towards the target object until the object fills a minimum
    of percentage (given by `good_view_percentage`) of the sensor's field of view or the
    closest point of the object is less than `desired_object_distance` from the sensor.
    This makes sure that big and small objects all fill similar amount of space in the
    sensor's field of view. Otherwise small objects may be too small to perform saccades
    or the sensor ends up inside of big objects. This step is performed by default but
    can be skipped by setting `allow_translation=False`.

    Second, the agent will then be oriented towards the object so that the sensor's
    central pixel is on-object. In the case of multi-object experiments,
    (i.e., when `multiple_objects_present=True`), there is an additional orientation
    step performed prior to the translational movement step.
    """

    def __init__(
        self,
        agent_id: AgentID,
        desired_object_distance: float,
        good_view_percentage: float,
        multiple_objects_present: bool,
        sensor_id: SensorID,
        target_semantic_id: SemanticID,
        allow_translation: bool = True,
        max_orientation_attempts: int = 1,
    ) -> None:
        """Initialize the GetGoodView positioning procedure.

        Args:
            agent_id: The ID of the agent to generate actions for.
            desired_object_distance: The desired distance to the object.
            good_view_percentage: The percentage of the sensor that should be
                filled with the object.
            multiple_objects_present: Whether there are multiple objects in
                the scene.
            sensor_id: The ID of the sensor to use for positioning.
            target_semantic_id: The semantic ID of the target object.
            allow_translation: Whether to allow movement toward the object via
                the motor systems's move_close_enough method. If False, only
                orientienting movements are performed. Defaults to True.
            max_orientation_attempts: The maximum number of orientation attempts
                allowed before giving up and truncating the procedure indicating that
                the sensor is not on the target object.
        """
        self._agent_id = agent_id
        self._desired_object_distance = desired_object_distance
        self._good_view_percentage = good_view_percentage
        self._multiple_objects_present = multiple_objects_present
        self._sensor_id = sensor_id
        self._target_semantic_id = target_semantic_id
        self._allow_translation = allow_translation
        self._max_orientation_attempts = max_orientation_attempts

        self._num_orientation_attempts = 0
        self._executed_multiple_objects_orientation = False

    def compute_look_amounts(
        self, relative_location: np.ndarray, state: MotorSystemState
    ) -> tuple[float, float]:
        """Compute the amount to look down and left given a relative location.

        This function computes the amount needed to look down and left in order
        for the sensor to be aimed at the target. The returned amounts are relative
        to the agent's current position and rotation. Looking up and right is done
        by returning negative amounts.

        TODO: Test whether this function works when the agent is facing in the
        positive z-direction. It may be fine, but there were some adjustments to
        accommodate the z-axis positive direction pointing opposite the body's initial
        orientation (e.g., using negative  `z` in
        `left_amount = -np.degrees(np.arctan2(x_rot, -z_rot)))`.

        Args:
            relative_location: the x,y,z coordinates of the target with respect
                to the sensor.
            state: The current state of the motor system. Defaults to None.

        Returns:
            down_amount: Amount to look down (degrees).
            left_amount: Amount to look left (degrees).
        """
        sensor_rotation_rel_world = self.sensor_rotation_relative_to_world(state)

        # Invert the sensor rotation and apply it to the relative location
        w, x, y, z = qt.as_float_array(sensor_rotation_rel_world)
        rotation = rot.from_quat([x, y, z, w])
        rotated_location = rotation.inv().apply(relative_location)

        # Calculate the necessary rotation amounts and convert them to degrees.
        x_rot, y_rot, z_rot = rotated_location
        left_amount = -np.degrees(np.arctan2(x_rot, -z_rot))
        distance_horiz = np.sqrt(x_rot**2 + z_rot**2)
        down_amount = -np.degrees(np.arctan2(y_rot, distance_horiz))

        return down_amount, left_amount

    def find_location_to_look_at(
        self,
        sem3d_obs: np.ndarray,
        image_shape: tuple[int, int],
        state: MotorSystemState,
    ) -> np.ndarray:
        """Find the location to look at in the observation.

        Takes in a semantic 3D observation and returns an x,y,z location.

        The location is on the object and surrounded by pixels that are also on
        the object. This is done by smoothing the on_object image and then
        taking the maximum of this smoothed image.

        Args:
            sem3d_obs: The location of each pixel and the semantic ID
                associated with that location.
            image_shape: The shape of the camera image.
            state: The current state of the motor system.

        Returns:
            The x,y,z coordinates of the target with respect to the sensor.
        """
        sem3d_obs_image = sem3d_obs.reshape((image_shape[0], image_shape[1], 4))
        on_object_image = sem3d_obs_image[:, :, 3]

        if not self._multiple_objects_present:
            on_object_image[on_object_image > 0] = self._target_semantic_id

        on_object_image = on_object_image == self._target_semantic_id
        on_object_image = on_object_image.astype(float)

        # TODO add unit test that we make sure find_location_to_look at functions
        # as expected, which can otherwise break if e.g. on_object_image is passed
        # as an int or boolean rather than float
        kernel_size = on_object_image.shape[0] // 16
        smoothed_on_object_image = scipy.ndimage.gaussian_filter(
            on_object_image, kernel_size, mode="constant"
        )
        idx_loc_to_look_at = np.argmax(smoothed_on_object_image * on_object_image)
        idx_loc_to_look_at = np.unravel_index(idx_loc_to_look_at, on_object_image.shape)
        location_to_look_at = sem3d_obs_image[
            idx_loc_to_look_at[0], idx_loc_to_look_at[1], :3
        ]
        camera_location = (
            state[self._agent_id].sensors[SensorID(f"{self._sensor_id}.depth")].position
        )
        agent_location = state[self._agent_id].position
        # Get the location of the object relative to sensor.
        return location_to_look_at - (camera_location + agent_location)

    def is_on_target_object(self, observation: Mapping) -> bool:
        """Check if a sensor is on the target object.

        Args:
            observation: The observation to use for positioning.

        Returns:
            Whether the sensor is on the target object.
        """
        # Reconstruct the 2D semantic/surface map embedded in 'semantic_3d'.
        image_shape = observation[self._agent_id][self._sensor_id]["depth"].shape[0:2]
        semantic_3d = observation[self._agent_id][self._sensor_id]["semantic_3d"]
        semantic = semantic_3d[:, 3].reshape(image_shape).astype(int)
        if not self._multiple_objects_present:
            semantic[semantic > 0] = self._target_semantic_id

        # Check if the central pixel is on the target object.
        y_mid, x_mid = image_shape[0] // 2, image_shape[1] // 2
        return semantic[y_mid, x_mid] == self._target_semantic_id

    def move_close_enough(self, observation: Mapping) -> Action | None:
        """Move closer to the object until we are close enough.

        Args:
            observation: The observation to use for positioning.

        Returns:
            The next action to take, or None if we are already close enough to the
            object.

        Raises:
            ValueError: If the object is not visible.
        """
        # Reconstruct 2D semantic map.
        depth_image = observation[self._agent_id][self._sensor_id]["depth"]
        semantic_3d = observation[self._agent_id][self._sensor_id]["semantic_3d"]
        semantic_image = semantic_3d[:, 3].reshape(depth_image.shape).astype(int)

        if not self._multiple_objects_present:
            semantic_image[semantic_image > 0] = self._target_semantic_id

        points_on_target_obj = semantic_image == self._target_semantic_id
        n_points_on_target_obj = points_on_target_obj.sum()

        # For multi-object experiments, handle the possibility that object is no
        # longer visible.
        if self._multiple_objects_present and n_points_on_target_obj == 0:
            logger.debug("Object not visible, cannot move closer")
            return None

        if n_points_on_target_obj > 0:
            closest_point_on_target_obj = np.min(depth_image[points_on_target_obj])
            logger.debug(
                "closest target object point: " + str(closest_point_on_target_obj)
            )
        else:
            raise ValueError(
                "May be initializing experiment with no visible target object"
            )

        perc_on_target_obj = get_perc_on_obj_semantic(
            semantic_image, semantic_id=self._target_semantic_id
        )
        logger.debug("% on target object: " + str(perc_on_target_obj))

        # Also calculate closest point on *any* object so that we don't get too close
        # and clip into objects; NB that any object will have a semantic ID > 0
        points_on_any_obj = semantic_image > 0
        closest_point_on_any_obj = np.min(depth_image[points_on_any_obj])
        logger.debug("closest point on any object: " + str(closest_point_on_any_obj))

        if perc_on_target_obj < self._good_view_percentage:
            if closest_point_on_target_obj > self._desired_object_distance:
                if self._multiple_objects_present and (
                    closest_point_on_any_obj < self._desired_object_distance / 4
                ):
                    logger.debug(
                        "Getting too close to other objects, not moving forward."
                    )
                    return None

                logger.debug("Moving forward")
                return MoveForward(agent_id=self._agent_id, distance=0.01)

            logger.debug("Close enough.")
            return None

        logger.debug("Enough percent visible.")
        return None

    def orient_to_object(
        self, observation: Mapping, state: MotorSystemState
    ) -> list[Action]:
        """Rotate sensors so that they are centered on the object using the view finder.

        The view finder needs to be in the same position as the sensor patch
        and the object needs to be somewhere in the view finders view.

        Args:
            observation: The observation to use for positioning.
            state: The current state of the motor system.

        Returns:
            A list of actions of length two composed of actions needed to get us onto
            the target object.
        """
        # Reconstruct 2D semantic map.
        depth_image = observation[self._agent_id][self._sensor_id]["depth"]
        obs_dim = depth_image.shape[0:2]
        sem3d_obs = observation[self._agent_id][self._sensor_id]["semantic_3d"]
        sem_obs = sem3d_obs[:, 3].reshape(obs_dim).astype(int)

        if not self._multiple_objects_present:
            sem_obs[sem_obs > 0] = self._target_semantic_id

        logger.debug("Searching for object")
        relative_location = self.find_location_to_look_at(
            sem3d_obs,
            image_shape=obs_dim,
            state=state,
        )
        down_amount, left_amount = self.compute_look_amounts(
            relative_location, state=state
        )
        return [
            LookDown(agent_id=self._agent_id, rotation_degrees=down_amount),
            TurnLeft(agent_id=self._agent_id, rotation_degrees=left_amount),
        ]

    def __call__(
        self,
        observation: Mapping,
        state: MotorSystemState,
    ) -> PositioningProcedureResult:
        if (
            self._multiple_objects_present
            and not self._executed_multiple_objects_orientation
        ):
            # Setting this flag to True here ensures that this state machine state is
            # not revisited.
            self._executed_multiple_objects_orientation = True
            on_target_object = self.is_on_target_object(observation)
            if not on_target_object:
                return PositioningProcedureResult(
                    actions=self.orient_to_object(observation, state)
                )

        if self._allow_translation:
            action = self.move_close_enough(observation)
            if action is not None:
                logger.debug("Moving closer to object.")
                return PositioningProcedureResult(actions=[action])

        # Setting this flag to False here ensures that this state machine state is
        # not revisited.
        self._allow_translation = False

        on_target_object = self.is_on_target_object(observation)
        if (
            not on_target_object
            and self._num_orientation_attempts < self._max_orientation_attempts
        ):
            self._num_orientation_attempts += 1
            return PositioningProcedureResult(
                actions=self.orient_to_object(observation, state)
            )

        if on_target_object:
            return PositioningProcedureResult(success=True, terminated=True)

        return PositioningProcedureResult(truncated=True)

    def sensor_rotation_relative_to_world(self, state: MotorSystemState) -> Any:
        """Derives the positioning sensor's rotation relative to the world.

        Args:
            state: The current state of the motor system.

        Returns:
            The positioning sensor's rotation relative to the world.
        """
        agent_state = state[self._agent_id]
        # Retrieve agent's rotation relative to the world.
        agent_rotation = agent_state.rotation
        # Retrieve sensor's rotation relative to the agent.
        sensor_rotation = agent_state.sensors[
            SensorID(f"{self._sensor_id}.depth")
        ].rotation
        # Derive sensor's rotation relative to the world.
        return agent_rotation * sensor_rotation
