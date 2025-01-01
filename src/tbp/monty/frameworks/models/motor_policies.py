# Copyright 2025 Thousand Brains Project
# Copyright 2021-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import abc
import copy
import json
import logging
import math
import os
from typing import Any, Callable, Dict, List, Tuple, Type, Union, cast

import numpy as np
import quaternion as qt
import scipy.ndimage
from scipy.spatial.transform import Rotation as rot  # noqa: N813

from tbp.monty.frameworks.actions.action_samplers import ActionSampler
from tbp.monty.frameworks.actions.actions import (
    Action,
    ActionJSONDecoder,
    ActionJSONEncoder,
    LookDown,
    LookUp,
    MoveForward,
    MoveTangentially,
    OrientHorizontal,
    OrientVertical,
    SetAgentPose,
    SetSensorRotation,
    TurnLeft,
    TurnRight,
    VectorXYZ,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import get_angle_beefed_up
from tbp.monty.frameworks.utils.transform_utils import scipy_to_numpy_quat


class MotorSystem(abc.ABC, Callable):
    @property
    @abc.abstractmethod
    def last_action(self) -> Action:
        pass

    @abc.abstractmethod
    def update(self, actions):
        # TODO: nobody is using this. Is this still valid?
        pass

    @abc.abstractmethod
    def set_experiment_mode(self, mode):
        pass

    @abc.abstractmethod
    def dynamic_call(self) -> Action:
        """Use this method when actions are not predefined."""
        pass

    @abc.abstractmethod
    def predefined_call(self) -> Tuple[Union[str, None], Any]:
        """Use this method when actions are not predefined."""
        pass

    @abc.abstractmethod
    def post_action(self, action: Action) -> None:
        """This method will automatically be called at the end of __call__."""
        pass

    def __call__(self) -> Action:
        """Defines the structure for __call__.

        Selects dynamic or predfined call, and then calls post_action.

        Returns:
            Action to take.
        """
        if self.is_predefined:
            action = self.predefined_call()
        else:
            action = self.dynamic_call()
        self.post_action(action)
        return action


class BasePolicy(MotorSystem):
    def __init__(
        self,
        rng,
        action_sampler_args: Dict,
        action_sampler_class: Type[ActionSampler],
        agent_id: str,
        switch_frequency,
        file_name=None,
        file_names_per_episode=None,
    ):
        """Initialize a base policy.

        Args:
            rng: Random number generator to use
            action_sampler_args: arguments for the ActionSampler
            action_sampler_class: The ActionSampler to use
            agent_id: The agent ID
            switch_frequency: float in [0,1], how frequently to change actions
                when using sticky actions
            file_name: Path to file with predefined actions. Defaults to None.
            file_names_per_episode: ?. Defaults to None.
        """
        ###
        # Define instance attributes
        ###
        self.rng = rng
        self.agent_id = agent_id

        self.action_sampler = action_sampler_class(rng=self.rng, **action_sampler_args)

        self.action_sequence = []
        self.timestep = 0
        self.episode_step = 0
        self.episode_count = 0
        self.switch_frequency = float(switch_frequency)
        # Ensure our first action only samples from those that can be random
        self.action = self.get_random_action(self.action_sampler.sample(self.agent_id))
        self.state = None

        ###
        # Load data for predefined actions and amounts if specified
        ###

        self.is_predefined = False
        self.file_names_per_episode = None
        self.action_list = []

        # Don't want to go around and change all uses of file_name, so this is argument
        # is in addition to, rather than in replacement of, file_name
        if file_names_per_episode is not None:
            self.file_names_per_episode = file_names_per_episode
            self.is_predefined = True

        if file_name is not None:
            self.action_list = read_action_file(file_name)
            self.is_predefined = True

    ###
    # Methods that define behavior of __call__
    ###

    def dynamic_call(self) -> Action:
        return self.get_random_action(self.action)

    def get_random_action(self, action: Action) -> Action:
        """Returns random action sampled from allowable actions.

        Enables expanding the action space of the base policy with actions that
        we don't necessarily want to randomly sample
        """
        while True:
            if self.rng.rand() < self.switch_frequency:
                action = self.action_sampler.sample(self.agent_id)
            if not isinstance(action, SetAgentPose) and not isinstance(
                action, SetSensorRotation
            ):
                return action

    def predefined_call(self) -> Action:
        return self.action_list[self.episode_step % len(self.action_list)]

    def post_action(self, action: Action) -> None:
        self.action = action
        self.timestep += 1
        self.episode_step += 1
        self.action_sequence.append([action])

    def pre_episode(self):
        self.episode_step = 0
        self.is_exploring = False
        self.action_sequence = []

    def post_episode(self):
        self.episode_count += 1
        if self.file_names_per_episode is not None:
            if self.episode_count in self.file_names_per_episode:
                file_name = self.file_names_per_episode[self.episode_count]
                self.action_list = read_action_file(file_name)

    ###
    # Other required abstract methods, methods called by Monty or Dataloader
    ###

    def get_agent_state(self):
        """Get agent state (dict).

        Note:
            Assumes we only have one agent.

        Returns:
            Agent state.
        """
        return self.state[self.agent_id]

    @property
    def is_motor_only_step(self):
        agent_state = self.get_agent_state()
        if "motor_only_step" in agent_state.keys() and agent_state["motor_only_step"]:
            return True
        else:
            return False

    def last_action(self) -> Action:
        return self.action

    def update(self, *args):
        pass

    def state_dict(self):
        return {"timestep": self.timestep, "episode_step": self.episode_step}

    def load_state_dict(self, state_dict):
        self.timestep = state_dict["timestep"]
        self.episode_step = state_dict["episode_step"]

    def set_experiment_mode(self, mode):
        pass


class JumpToGoalStateMixin:
    """Convert driving goal state to an action in Habitat-compatible coordinates.

    Motor policy that enables us to take in a driving goal state for the motor agent,
    and specify the action in Habitat-compatible coordinates that must be taken
    to move there.
    """

    def __init__(self) -> None:
        self.driving_goal_state = None

    def pre_episode(self):
        self.set_driving_goal_state(None)

    def set_driving_goal_state(self, goal_state):
        """Specify the goal-state that the motor-actuator will attempt to satisfy."""
        self.driving_goal_state = goal_state

    def derive_habitat_goal_state(self):
        """Derive the Habitat-compatible goal state.

        Take the current driving goal state (in CMP format), and derive the
        corresponding Habitat compatible goal-state to pass through the Embodied
        Dataloader.

        Returns:
            target_loc: Target location.
            target_quat: Target quaternion.
        """
        if self.driving_goal_state is not None:
            target_loc = self.driving_goal_state.location
            target_agent_vec = self.driving_goal_state.morphological_features[
                "pose_vectors"
            ][0]

            yaw_angle = math.atan2(-target_agent_vec[0], -target_agent_vec[2])
            pitch_angle = math.asin(target_agent_vec[1])

            # Should rotate by pitch degrees around x, and by yaw degrees around y (and
            # no change about z, which would correspond to roll)
            scipy_combined_orientation = rot.from_euler(
                "xyz",
                [pitch_angle, yaw_angle, 0],
                degrees=False,
            )

            target_quat = scipy_to_numpy_quat(scipy_combined_orientation.as_quat())

            # Reset driving goal state and await further inputs
            self.set_driving_goal_state(None)

            return target_loc, target_quat

        else:
            return None, None


class InformedPolicy(BasePolicy, JumpToGoalStateMixin):
    """Policy that takes observation as input.

    Extension of BasePolicy that allows for taking the observation into account for
    action selection. Currently it uses the percentage of the observation that is on
    the object to reverse the last action if it is below min_perc_on_obj.

    Additionally, this policy discouraces taking the reverse of the previous action
    if we are still on the object.

    Attributes:
        guiding_sensors: List of sensors that are used to calculate the percentage
            on object. When using multiple sensors or a visualization sensor we may
            want to ignore some when determining whether we need to move back.
        min_perc_on_obj: How much percent of the observation needs to be on the
            object to sample a new action. Otherwise the previous action is reversed to
            get back on the object. TODO: Not used anywhere?
    """

    def __init__(
        self,
        min_perc_on_obj,
        good_view_percentage,
        desired_object_distance,
        use_goal_state_driven_actions=False,
        **kwargs,
    ):
        """Initialize policy.

        Args:
            min_perc_on_obj: Minimum percentage of patch that needs to be on the object.
                If under this amount, reverse the previous action to get the patch back
                on the object.
            good_view_percentage: How much percent of the view finder perception should
                be filled with the object. (If less, move closer)
            desired_object_distance: How far away should the agent be from the object
                in view; for the distant-agent, this is used to establish a maximum
                allowable distance of the object; note for the surface agent, this is
                used with every set of traversal steps to ensure we remain close to the
                surface
            use_goal_state_driven_actions: Whether to enable the motor system to make
                use of the JumpToGoalStateMixin, which attempts to "jump" (i.e.
                teleport) the agent to a specified goal state.
            **kwargs: Additional keyword arguments.
        """
        super(InformedPolicy, self).__init__(**kwargs)
        self.min_perc_on_obj = min_perc_on_obj
        self.good_view_percentage = good_view_percentage
        self.desired_object_distance = desired_object_distance
        self.use_goal_state_driven_actions = use_goal_state_driven_actions
        if self.use_goal_state_driven_actions:
            JumpToGoalStateMixin.__init__(self)

        # Observations after passing through sensor modules.
        # Are updated in Monty step method.
        self.processed_observations = None

    def pre_episode(self):
        if self.use_goal_state_driven_actions:
            JumpToGoalStateMixin.pre_episode(self)

        return super().pre_episode()

    def get_depth_at_center(self, raw_observation, view_sensor_id, initial_pose=True):
        """Determine the depth of the central pixel.

        Method primarily used by surface-agent, but also by distant agent after
        performing a hypothesis-testing jump; determines the depth of the central pixel,
        to inform whether the object is visible at all

        initial_pose : Whether we are checking depth-at center as part of the first
            start of an experiment; if using get_depth_at_center to check for
            the observation after e.g. a hypothesis-testing jump, then we don't
            want to throw an error if the object is nowhere to be seen; instead, we
            simply move back

        Returns:
            Depth at the center of the view sensor.
        """
        observation_shape = raw_observation[self.agent_id][view_sensor_id][
            "depth"
        ].shape
        depth_at_center = raw_observation[self.agent_id][view_sensor_id]["depth"][
            observation_shape[0] // 2, observation_shape[1] // 2
        ]
        if initial_pose:
            assert (
                depth_at_center > 0
            ), "Object must be initialized such that the agent can visualize it by moving forward"  # noqa: E501
            # TODO investigate - I think this may have always been passing in the
            # original surface-agent policy implementation because the surface
            # sensor clips at 1.0, so even if the object isn't strictly visible (or
            # we're inside the object))  there's a risk we have initialized without the
            # object in view
        return depth_at_center

    ###
    # Methods that define behavior of __call__
    ###

    def dynamic_call(self) -> Action:
        """Return the next action and amount.

        This requires self.processed_observations to be updated at every step
        in the Monty class. self.processed_observations contains the features
        extracted by the sensor module for the guiding sensor (patch).
        """
        return (
            super().dynamic_call()
            if self.processed_observations.get_on_object()
            else self.fixme_undo_last_action()
        )

    def fixme_undo_last_action(self):
        """Returns an action that undoes last action for supported actions.

        Previous InformedPolicy.dynamic_call() implementation when not on object:

            action, amount = (last_action, -last_amount)

        This implementation duplicates the functionality and the implicit
        assumption in the code and configurations that InformedPolicy is working
        with one of the following actions:
        - LookUp
        - LookDown
        - TurnLeft
        - TurnRight

        Additionally, this implementation adds support for:
        - MoveForward
        - MoveTangentially

        Additional support for the above two actions is due to `-last_amount`
        working for these actions as well. This maintains the same code functionality
        during this refactoring.

        For other actions, raise ValueError explicitly.

        Raises:
            ValueError: If the last action is not supported

        TODO These instance checks are undesirable and should be removed in the future.
        I am using these for now to express the implicit assumptions in the code.
        An Action.undo of some sort would be a better solution, however it is not
        yet clear to me what to do for actions that do not support undo.
        """
        last_action = self.last_action()

        if isinstance(last_action, LookDown):
            return LookDown(
                agent_id=last_action.agent_id,
                rotation_degrees=-last_action.rotation_degrees,
                constraint_degrees=last_action.constraint_degrees,
            )
        elif isinstance(last_action, LookUp):
            return LookUp(
                agent_id=last_action.agent_id,
                rotation_degrees=-last_action.rotation_degrees,
                constraint_degrees=last_action.constraint_degrees,
            )
        elif isinstance(last_action, TurnLeft):
            return TurnLeft(
                agent_id=last_action.agent_id,
                rotation_degrees=-last_action.rotation_degrees,
            )
        elif isinstance(last_action, TurnRight):
            return TurnRight(
                agent_id=last_action.agent_id,
                rotation_degrees=-last_action.rotation_degrees,
            )
        elif isinstance(last_action, MoveForward):
            return MoveForward(
                agent_id=last_action.agent_id,
                distance=-last_action.distance,
            )
        elif isinstance(last_action, MoveTangentially):
            return MoveTangentially(
                agent_id=last_action.agent_id,
                distance=-last_action.distance,
                # Same direction, negative distance
                direction=last_action.direction,
            )
        else:
            raise ValueError(f"Invalid action: {last_action}")

    def post_action(self, action: Action) -> None:
        self.action = action
        self.timestep += 1
        self.episode_step += 1
        if self.is_exploring is False:
            state_copy = self.convert_motor_state()
            self.action_sequence.append([action, state_copy])

    def convert_motor_state(self):
        """Convert the motor state into something that can be pickled/saved to JSON.

        i.e. substitute vector and quaternion objects; note e.g. copy.deepcopy does not
        work.

        TODO ?clean this up with a recursive algorithm, or use BufferEncoder in
        buffer.py

        Returns:
            Copy of the agent state.
        """
        state_copy = dict()
        for key in self.state.keys():
            state_copy[key] = dict()
            for key_inner in self.state[key].keys():
                if type(self.state[key][key_inner]) is dict:
                    state_copy[key][key_inner] = dict()
                    # We need to go deeper
                    for key_inner_inner in self.state[key][key_inner].keys():
                        state_copy[key][key_inner][key_inner_inner] = dict()
                        if type(self.state[key][key_inner][key_inner_inner]) is dict:
                            # We need to go even deeper...
                            # (**Hans Zimmer music intensifies**)
                            for key_i_i_i in self.state[key][key_inner][
                                key_inner_inner
                            ]:
                                state_copy[key][key_inner][key_inner_inner][
                                    key_i_i_i
                                ] = dict()
                                try:
                                    state_copy[key][key_inner][key_inner_inner][
                                        key_i_i_i
                                    ] = np.array(
                                        [
                                            x
                                            for x in self.state[key][key_inner][
                                                key_inner_inner
                                            ][key_i_i_i]
                                        ]
                                    )
                                except TypeError:
                                    # Quaternions
                                    state_copy[key][key_inner][key_inner_inner][
                                        key_i_i_i
                                    ] = [
                                        self.state[key][key_inner][key_inner_inner][
                                            key_i_i_i
                                        ].real
                                    ] + [
                                        x
                                        for x in self.state[key][key_inner][
                                            key_inner_inner
                                        ][key_i_i_i].imag
                                    ]
                elif type(self.state[key][key_inner]) is bool:
                    pass
                else:
                    try:
                        state_copy[key][key_inner] = np.array(
                            [x for x in self.state[key][key_inner]]
                        )
                    except TypeError:
                        # Quaternions
                        state_copy[key][key_inner] = [
                            self.state[key][key_inner].real
                        ] + [x for x in self.state[key][key_inner].imag]
        return state_copy

    ###
    # Methods called by Dataloader and associated helper methods
    ###

    def move_close_enough(
        self, raw_observation, view_sensor_id, target_semantic_id, multi_objects_present
    ) -> Tuple[Union[Action, None], bool]:
        """At beginning of episode move close enough to the object.

        Used the raw observations returned from the dataloader and not the
        extracted features from the sensor module.

        Args:
            raw_observation: The raw observations from the dataloader
            view_sensor_id: The ID of the view sensor
            target_semantic_id: The semantic ID of the primary target object in the
                scene.
            multi_objects_present: Whether there are multiple objects present in the
                scene. If so, we do additional checks to make sure we don't get too
                close to these when moving forward

        Returns:
            Tuple[Union[Action, None], bool]: The next action to take and whether the
                episode is done

        Raises:
            ValueError: If the object is not visible
        """
        view = raw_observation[self.agent_id][view_sensor_id]["semantic"]
        points_on_target_obj = (
            raw_observation[self.agent_id][view_sensor_id]["semantic"]
            == target_semantic_id
        )

        # For multi-object experiments, handle the possibility that object is no
        # longer visible
        if multi_objects_present and (
            len(
                raw_observation[self.agent_id][view_sensor_id]["depth"][
                    points_on_target_obj
                ]
            )
            == 0
        ):
            logging.debug("Object not visible, cannot move closer")
            return None, True

        if len(points_on_target_obj) > 0:
            closest_point_on_target_obj = np.min(
                raw_observation[self.agent_id][view_sensor_id]["depth"][
                    points_on_target_obj
                ]
            )
            logging.debug(
                "closest target object point: " + str(closest_point_on_target_obj)
            )
        else:
            raise ValueError(
                "May be initializing experiment with no visible target object"
            )

        perc_on_target_obj = get_perc_on_obj_semantic(
            view, sematic_id=target_semantic_id
        )
        logging.debug("% on target object: " + str(perc_on_target_obj))

        # Also calculate closest point on *any* object so that we don't get too close
        # and clip into objects; NB that any object will have a semantic ID > 0
        points_on_any_obj = (
            raw_observation[self.agent_id][view_sensor_id]["semantic"] > 0
        )
        closest_point_on_any_obj = np.min(
            raw_observation[self.agent_id][view_sensor_id]["depth"][points_on_any_obj]
        )
        logging.debug("closest point on any object: " + str(closest_point_on_any_obj))

        if perc_on_target_obj < self.good_view_percentage:
            if closest_point_on_target_obj > self.desired_object_distance:
                if multi_objects_present and (
                    closest_point_on_any_obj < self.desired_object_distance / 4
                ):
                    logging.debug(
                        "getting too close to other objects, not moving forward"
                    )
                    return None, True  # done
                else:
                    logging.debug("move forward")
                    return MoveForward(agent_id=self.agent_id, distance=0.01), False
            else:
                logging.debug("close enough")
                return None, True  # done
        else:
            logging.debug("Enough percent visible")
            return None, True  # done

    def orient_to_object(
        self, raw_observation, view_sensor_id, target_semantic_id
    ) -> Tuple[List[Action], bool]:
        """Rotate sensors so that they are centered on the object using a view finder.

        The view finder needs to be in the same position as the sensor patch
        and the object needs to be somewhere in the view finders view.

        Args:
            raw_observation: raw observations of the view finder
            view_sensor_id: view finder id (str)
            target_semantic_id: the integer corresponding to the semantic ID
                of the target object that we will try to fixate on

        Returns:
            Two actions to execute to put the patch on the object
        """
        sem_obs = raw_observation[self.agent_id][view_sensor_id]["semantic"]
        sem3d_obs = raw_observation[self.agent_id][view_sensor_id]["semantic_3d"]
        obs_dim = sem_obs.shape

        logging.debug("Searching for object")

        # Check if the central pixel is on-object.
        y_mid, x_mid = obs_dim[0] // 2, obs_dim[1] // 2
        if sem_obs[y_mid, x_mid] == target_semantic_id:
            logging.debug("Already centered on the object")
            return [], True

        relative_location = self.find_location_to_look_at(
            sem3d_obs, image_shape=obs_dim, target_semantic_id=target_semantic_id
        )
        down_amount, left_amount = self.compute_look_amounts(relative_location)

        return (
            [
                LookDown(agent_id=self.agent_id, rotation_degrees=down_amount),
                TurnLeft(agent_id=self.agent_id, rotation_degrees=left_amount),
            ],
            False,
        )

    def compute_look_amounts(self, relative_location):
        """Compute the amount to look down and left given a relative location.

        TODO: rotate/translate the relative location so that this works regardless
        of the camera location and rotation.

        Returns:
            down_amount: Amount to look down.
            left_amount: Amount to look left.
        """
        down_amount = np.degrees(np.arctan2(relative_location[1], relative_location[2]))
        left_amount = np.degrees(np.arctan2(relative_location[0], relative_location[2]))
        return down_amount, left_amount

    def find_location_to_look_at(self, sem3d_obs, image_shape, target_semantic_id):
        """Takes in a semantic 3D observation and returns an x,y,z location.

        The location is on the object and surrounded by pixels that are also on
        the object. This is done by smoothing the on_object image and then
        taking the maximum of this smoothed image

        Args:
            sem3d_obs: the location of each pixel and the semantic ID associated
                with that location
            image_shape: the shape of the camera image
            target_semantic_id: the semantic ID of the target object we'd like to
                saccade on to

        Returns:
            relative_location: the x,y,z distance from camera to pixel with max
                smoothed on_object value
        """
        sem3d_obs_image = sem3d_obs.reshape((image_shape[0], image_shape[1], 4))
        on_object_image = sem3d_obs_image[:, :, 3]

        on_object_image = on_object_image == target_semantic_id
        on_object_image = on_object_image.astype(float)

        # TODO add unit test that we make sure find_location_to_look at functions
        # as expected, which can otherwise break if e.g. on_object_image is passed
        # as an int or boolean rather than float
        smoothed_on_object_image = scipy.ndimage.gaussian_filter(
            on_object_image, 2, mode="constant"
        )
        idx_loc_to_look_at = np.argmax(smoothed_on_object_image * on_object_image)
        idx_loc_to_look_at = np.unravel_index(idx_loc_to_look_at, on_object_image.shape)
        location_to_look_at = sem3d_obs_image[
            idx_loc_to_look_at[0], idx_loc_to_look_at[1], :3
        ]
        camera_location = self.get_agent_state()["sensors"]["view_finder.depth"][
            "position"
        ]
        agent_location = self.get_agent_state()["position"]
        relative_location = (camera_location + agent_location) - location_to_look_at
        return relative_location

    def get_sensors_perc_on_obj(self, observation):
        """Calculate how much percent of the sensor is on the object.

        Get the average percentage of pixels on the object for all sensors in the
        list of guiding sensors.

        Returns:
            perc_on_obj: Percentage of pixels on the object.
        """
        perc_on_obj = 0
        # Currently assumes that we only have one agent.
        for sensor_id in self.guiding_sensors:
            sensor_obs = observation[self.agent_id][sensor_id]["rgba"]
            perc_on_obj += get_perc_on_obj(sensor_obs) / len(self.guiding_sensors)
        return perc_on_obj


class NaiveScanPolicy(InformedPolicy):
    """Policy that just moves left and right along the object."""

    def __init__(
        self,
        fixed_amount,
        **kwargs,
    ):
        """Initialize policy."""
        # Mostly use version of InformedPolicy to get the good view in the beginning
        # TODO: maybe separate this out. Don't need to specify reverse_actions or
        # min_perc_on_obj for that.
        super(NaiveScanPolicy, self).__init__(**kwargs)

        # Specify this specific action space, otherwise it doesn't work
        self._naive_scan_actions = [
            LookUp(agent_id=self.agent_id, rotation_degrees=fixed_amount),
            TurnLeft(agent_id=self.agent_id, rotation_degrees=fixed_amount),
            LookDown(agent_id=self.agent_id, rotation_degrees=fixed_amount),
            TurnRight(agent_id=self.agent_id, rotation_degrees=fixed_amount),
        ]
        self.fixed_amount = fixed_amount
        self.steps_per_action = 1
        self.current_action_id = 0
        self.step_on_action = 0

    ###
    # Methods that define behavior of __call__
    ###

    def dynamic_call(self) -> Action:
        if self.steps_per_action * self.fixed_amount >= 90:
            # Raise "StopIteration" to notify the dataloader we need to stop
            # the experiment. This exception is automatically handled by any
            # python loop statements using iterators.
            # See https://docs.python.org/3/library/exceptions.html#StopIteration
            raise StopIteration()
        else:
            self.check_cycle_action()
            action = self._naive_scan_actions[self.current_action_id]
        self.step_on_action += 1
        return action

    def pre_episode(self):
        super().pre_episode()
        self.steps_per_action = 1
        self.current_action_id = 0
        self.step_on_action = 0

    def check_cycle_action(self):
        """Makes sure we move in a spiral.

        This method switches the current action if steps_per_action was reached.
        Additionally it increments steps_per_action after the second and forth action
        to make sure paths don't overlap.
         _ _ _ _
        |  _ _  |
        | |_  | |
        |_ _ _| |

        corresponds to
        1x left, 1x up,
        2x right, 2x down,
        3x left, 3x up,
        4x right, 4x down,
        ...
        """
        if self.step_on_action == self.steps_per_action:
            self.current_action_id += 1
            self.step_on_action = 0
            if self.current_action_id == 2:
                self.steps_per_action += 1
            elif self.current_action_id == 4:
                self.current_action_id = 0
                self.steps_per_action += 1


class SurfacePolicy(InformedPolicy):
    """Policy class for a surface-agent.

    i.e. an agent that moves to and follows the surface of an object. Includes
    functions for moving along an object based on its surface normal.
    """

    def __init__(
        self,
        alpha,
        min_perc_on_obj=0.25,
        good_view_percentage=0.5,
        **kwargs,
    ):
        """Initialize policy.

        Args:
            min_perc_on_obj: Minimum percentage of patch that needs to be
                on the object. If under this amount, reverse the previous action
                to get the patch back on the object.
            good_view_percentage: How much percent of the view finder perception
                should be filled with the object. (If less, move closer)
                TODO M : since surface agent does not use get_good_view, can consider
                removing this parameter
            alpha: to what degree should the move_tangentially direction be the
                same as the last step or totally random? 0~same as before, 1~random walk
            **kwargs: ?
        """
        super().__init__(min_perc_on_obj, good_view_percentage, **kwargs)
        self.action = None
        self.tangential_angle = 0
        self.alpha = alpha

    def pre_episode(self):
        self.tangential_angle = 0
        self.action = None  # Reset the first action for every episode
        self.touch_search_amount = 0  # Track how many rotations the agent has made
        # along the horizontal plane searching for an object; when this reaches 360,
        # try searching along the vertical plane, or for 720, performing a random
        # search

        return super().pre_episode()

    def touch_object(self, raw_observation, view_sensor_id) -> Action:
        """The surface agent's policy for moving onto an object for sensing it.

        Like the distant agent's get_good_view, this is called at the beginning
        of every episode, and after a "jump" has been initialized by a
        model-based policy. In addition, it can be called when the surface agent
        cannot sense the object, e.g. because it has fallen off its surface.

        Currently uses the raw observations returned from the viewfinder via the
        dataloader, and not the extracted features from the sensor module.
        TODO M refactor this so that all sensory processing is done in the sensor
        module.

        If we aren't on the object, try first systematically orienting left around
        a point, then orienting down, and finally random orientations along the surface
        of a fixed sphere.

        Returns:
            Action to take.
        """
        # If the viewfinder sees the object within range, then move to it
        depth_at_center = self.get_depth_at_center(raw_observation, view_sensor_id)
        if depth_at_center < 1.0:
            distance = (
                depth_at_center
                - self.desired_object_distance
                - self.state["agent_id_0"]["sensors"][f"{view_sensor_id}.depth"][
                    "position"
                ][2]
            )
            logging.debug(f"Move to touch visible object, forward by {distance}")

            self.action = MoveForward(agent_id=self.agent_id, distance=distance)
            return self.action

        logging.debug("Surface policy searching for object...")

        # Helpful to conceptualize these movements by considering a unit circle,
        # scaled by the radius distance_from_center
        # This image may be useful for intuition:
        # https://en.wikipedia.org/wiki/Exsecant#/media/File:Circle-trig6.svg

        # Rotate about a circle centered in fron of the agent's current
        # position; TODO decide how to deal with "coliding with"/entering an
        # object; ?could just rotate about a point on which the sensor is present
        # Currently as a heuristic we rotate about a point 4x the desired distance;
        # as this is typically 2.5cm, this would mean a circle with radius 10cm
        distance_from_center = self.desired_object_distance * 4

        rotation_degrees = 30  # 30 degrees at a time; note this amount is also used
        # to eventually re-orient ourselves back to the original central point;
        # TODO may want to consider trying smaller step-sizes; will
        # be less efficient, but may be important for smaller/distant objects that
        # otherwise get missed

        if self.touch_search_amount >= 720:
            # Perform a random upward or downward movement along the surface of a
            # sphere, with its centre fixed 10 cm in front of the agent
            logging.debug("Trying random search for object")
            if self.rng.uniform() < 0.5:
                orientation = "vertical"
                logging.debug("Orienting vertically")
            else:
                orientation = "horizontal"
                logging.debug("Orienting horizontally")

            rotation_degrees = self.rng.uniform(-180, 180)
            logging.debug(f"Random orientation amount is : {rotation_degrees}")

        elif self.touch_search_amount >= 360 and self.touch_search_amount < 720:
            logging.debug("Trying vertical search for object")
            orientation = "vertical"

        else:
            logging.debug("Trying horizontal search for object")
            orientation = "horizontal"

        # Move tangentally to the point we're facing, resulting in the agent's
        # orienation about the central point changing as specified by rotation_deg,
        # but now where the agent is no longer on the edge of the circle
        move_lat_amount = np.tan(np.radians(rotation_degrees)) * distance_from_center
        # The lateral movement will be down relative to the agent's facing position
        # in the case of orient vertical, and left in the case of orient horizontal

        # The below calculates the exsecant, which provides the necessary
        # movement to bring the agent back to the same radius around the central
        # point
        move_forward_amount = distance_from_center * (
            (1 - np.cos(np.radians(rotation_degrees)))
            / np.cos(np.radians(rotation_degrees))
        )

        self.touch_search_amount += rotation_degrees  # Accumulate total rotations
        # for touch-search

        if orientation == "vertical":
            self.action = OrientVertical(
                agent_id=self.agent_id,
                rotation_degrees=rotation_degrees,
                down_distance=move_lat_amount,
                forward_distance=move_forward_amount,
            )
        else:
            self.action = OrientHorizontal(
                agent_id=self.agent_id,
                rotation_degrees=rotation_degrees,
                left_distance=move_lat_amount,
                forward_distance=move_forward_amount,
            )
        return self.action

    ###
    # Methods that define behavior of __call__
    ###
    def dynamic_call(self) -> Action:
        """Return the next action to take.

        This requires self.processed_observations to be updated at every step
        in the Monty class. self.processed_observations contains the features
        extracted by the sensor module for the guiding sensor (patch).

        Returns:
            Action to take.
        """
        # Check if we have poor visualization of the object
        if self.processed_observations.get_feature_by_name("object_coverage") < 0.1:
            logging.debug(
                "Object coverage of only "
                + str(
                    self.processed_observations.get_feature_by_name("object_coverage")
                )
            )
            logging.debug("Initiating attempts to touch object")

            return None  # Will result in moving to try to find the object
            # This is determined by some logic in embodied_data.py, in particular
            # the next method of InformedEnvironmentDataLoader

        elif self.action is None:
            logging.debug(
                "Object coverage good at initialization: "
                + str(
                    self.processed_observations.get_feature_by_name("object_coverage")
                )
            )

            # In this case, we are on the first action, but the object view is already
            # good; therefore initialize the cycle of actions as if we had just
            # moved forward (e.g. to get a good view)
            self.action = self.action_sampler.sample_move_forward(self.agent_id)

        return self.get_next_action()

    def _orient_horizontal(self) -> Action:
        """Orient the agent horizontally.

        Returns:
            OrientHorizontal action.
        """
        rotation_degrees = self.orienting_angle_from_normal("horizontal")
        left_distance, forward_distance = self.horizontal_distances(rotation_degrees)
        return OrientHorizontal(
            agent_id=self.agent_id,
            rotation_degrees=rotation_degrees,
            left_distance=left_distance,
            forward_distance=forward_distance,
        )

    def _orient_vertical(self) -> Action:
        """Orient the agent vertically.

        Returns:
            OrientVertical action.
        """
        rotation_degrees = self.orienting_angle_from_normal("vertical")
        down_distance, forward_distance = self.vertical_distances(rotation_degrees)
        return OrientVertical(
            agent_id=self.agent_id,
            rotation_degrees=rotation_degrees,
            down_distance=down_distance,
            forward_distance=forward_distance,
        )

    def _move_tangentially(self) -> Action:
        """Move tangentially along the object surface.

        Returns:
            MoveTangentially action.
        """
        action = self.action_sampler.sample_move_tangentially(self.agent_id)

        # be careful if you're falling off the object!
        if self.processed_observations.get_feature_by_name("object_coverage") < 0.2:
            # Scale the step size by how small the object coverage is
            # Reduces situations where e.g. change in sensor resolution causes agent
            # to fall off the object
            action.distance = action.distance / (
                4 / self.processed_observations.get_feature_by_name("object_coverage")
            )
            logging.debug(f"Very close to edge so only moving by {action.distance}")

        elif self.processed_observations.get_feature_by_name("object_coverage") < 0.75:
            action.distance = action.distance / 4
            logging.debug(f"Near edge so only moving by {action.distance}")

        action.direction = self.tangential_direction()

        return action

    def _move_forward(self) -> Action:
        """Move forward to touch the object at the right distance.

        Returns:
            MoveForward action.
        """
        action = MoveForward(
            agent_id=self.agent_id,
            distance=(
                self.processed_observations.get_feature_by_name("min_depth")
                - self.desired_object_distance
            ),
        )
        return action

    def get_next_action(self):
        """Retrieve next action from a cycle of four actions.

        First move forward to touch the object at the right distance
        Then orient toward the normal along direction 1
        Then orient toward the normal along direction 2
        Then move tangentially along the object surface
        Then start over

        Returns:
            Next action in the cycle.
        """
        # TODO: is this check necessary?
        if not hasattr(self, "processed_observations"):
            return None

        last_action = self.last_action()

        if isinstance(last_action, MoveForward):
            return self._orient_horizontal()
        elif isinstance(last_action, OrientHorizontal):
            return self._orient_vertical()
        elif isinstance(last_action, OrientVertical):
            return self._move_tangentially()
        elif isinstance(last_action, MoveTangentially):
            # orient around object if it's not centered in view
            if not self.processed_observations.get_on_object():
                return self._orient_horizontal()
            # move to the desired_object_distance if it is in view
            else:
                return self._move_forward()

    def tangential_direction(self) -> VectorXYZ:
        """Set the direction of the action to be a direction 0 - 2pi.

        - start at 0 (go up) in the reference frame of the agent; i.e. based on
        the standard initialization of an agent, this will be up from the floor.
        To implement this convention, the theta is offset by 90 degrees when
        finding our x and y translations, i.e. such that theta of 0 results in
        moving up by 1 (y), and right by 0 (x), rather than vice-versa
        - random action -pi - +pi is given by (rand() - 0.5) * 2pi
        - These are combined and weighted by the alpha parameter

        Returns:
            VectorXYZ: direction of the action
        """
        new_target_direction = (self.rng.rand() - 0.5) * 2 * np.pi
        self.tangential_angle = (
            self.tangential_angle * (1 - self.alpha) + new_target_direction * self.alpha
        )

        direction = qt.rotate_vectors(
            self.state[self.agent_id]["rotation"],
            [
                np.cos(self.tangential_angle - np.pi / 2),
                np.sin(self.tangential_angle + np.pi / 2),
                0,
            ],
            # NB the np.pi/2 offset implements the convention that theta=0 should
            # result in moving up (i.e. along y), rather than right (i.e. along x),
            # while maintaing the standard association of cos(theta) with the
            # x-coordinate and sin(theta) with the y-coordinate of space
        )

        return direction

    def horizontal_distances(self, rotation_degrees: float) -> Tuple[float, float]:
        """Compute the horizontal and forward distances to move to.

        Compensate for a given rotation of a certain angle.

        Args:
            rotation_degrees: The angle to rotate by

        Returns:
            move_left_distance: The left distance to move
            move_forward_distance: The forward distance to move
        """
        rotation_radians = np.radians(rotation_degrees)
        depth = self.processed_observations.non_morphological_features["mean_depth"]

        move_left_distance = np.tan(rotation_radians) * depth
        move_forward_distance = (
            depth * (1 - np.cos(rotation_radians)) / np.cos(rotation_radians)
        )

        return move_left_distance, move_forward_distance

    def vertical_distances(self, rotation_degrees: float) -> Tuple[float, float]:
        """Compute the down and forward distances to move to.

        Compensate for a given rotation of a certain angle.

        Args:
            rotation_degrees: The angle to rotate by

        Returns:
            move_down_distance: The down distance to move
            move_forward_distance: The forward distance to move
        """
        rotation_radians = np.radians(rotation_degrees)
        depth = self.processed_observations.non_morphological_features["mean_depth"]

        move_down_distance = np.tan(rotation_radians) * depth
        move_forward_distance = (
            depth * (1 - np.cos(rotation_radians)) / np.cos(rotation_radians)
        )

        return move_down_distance, move_forward_distance

    def get_inverse_agent_rot(self):
        """Get the inverse rotation of the agent's current orientation.

        Used to transform poses of e.g. point normals or principle curvature from
        global coordinates into the coordinate frame of the agent.

        To intuit why we apply the inverse, imagine an e.g. point normal with the same
        pose as the agent; in the agent's reference frame, this should have the
        identity pose, which will be aquired by transforming the original pose by the
        inverse

        Returns:
            Inverse quaternion rotation.
        """
        # Note that quaternion format is [w, x, y, z]
        [w, x, y, z] = qt.as_float_array(self.state[self.agent_id]["rotation"])
        # Note that scipy.spatial.transform.Rotation (v1.10.0) format is [x, y, z, w]
        [x, y, z, w] = rot.from_quat([x, y, z, w]).inv().as_quat()
        return qt.quaternion(w, x, y, z)

    def orienting_angle_from_normal(self, orienting: str) -> float:
        """Compute turn angle to face the object.

        Based on the point normal, compute the angle that the agent needs
        to turn in order to be oriented directly toward the object

        Args:
            orienting: `"horizontal" or "vertical"`

        Returns:
            degrees that the agent needs to turn
        """
        original_point_normal = self.processed_observations.get_point_normal()

        inverse_quaternion_rotation = self.get_inverse_agent_rot()

        rotated_point_normal = qt.rotate_vectors(
            inverse_quaternion_rotation, original_point_normal
        )
        x, y, z = rotated_point_normal

        if "horizontal" == orienting:
            return -np.degrees(np.arctan(x / z)) if z != 0 else -np.sign(x)*90.0
        if "vertical" == orienting:
            return -np.degrees(np.arctan(y / z)) if z != 0 else -np.sign(y)*90.0


###
# Helper functions that can be used by multiple classes
###


def read_action_file(file: str) -> List[Action]:
    """Load a file with one action per line.

    Args:
        file: name of file to load

    Returns:
        List[Action]: list of actions
    """
    file = os.path.expanduser(file)
    with open(file, "r") as f:
        file_read = f.read()

    lines = [line.strip() for line in file_read.split("\n") if line.strip()]
    actions = [cast(Action, json.loads(line, cls=ActionJSONDecoder)) for line in lines]

    return actions


def write_action_file(actions: List[Action], file: str) -> None:
    """Write a list of actions to a file, one per line.

    Should be readable by read_action_file.

    Args:
        actions: list of actions
        file: path to file to save actions to
    """
    with open(file, "w") as f:
        for action in actions:
            f.write(f"{json.dumps(action, cls=ActionJSONEncoder)}\n")


def get_perc_on_obj(rgba_obs):
    """Get the percentage of pixels in the observation that land on the object.

    This uses the RGBA image which means it only works with one object in an
    empty void. With multiple objects this would need to be solved differently.

    Args:
        rgba_obs: RGBA image observation.

    Returns:
        perc_on_obj: Percentage of pixels on the object.
    """
    res = rgba_obs.shape[0]
    flat_obs = rgba_obs.reshape((res * res, 4))
    csum = np.sum(flat_obs[:, :3], axis=1)
    per_on_obj = np.sum(csum > 0) / (res * res)
    return per_on_obj


def get_perc_on_obj_semantic(semantic_obs, sematic_id=0):
    """Get the percentage of pixels in the observation that land on the target object.

    If a semantic ID is provided, then only pixels on the target object are counted;
    otherwise, pixels on any object are counted.

    This uses the semantic image, where each pixel is associated with a semantic ID
    that is unique for each object, and always >0.

    Args:
        semantic_obs: Semantic image observation.
        sematic_id: Semantic ID of the target object.

    Returns:
        perc_on_obj: Percentage of pixels on the object.
    """
    res = semantic_obs.shape[0] * semantic_obs.shape[1]
    if sematic_id == 0:
        csum = np.sum(semantic_obs >= 1)
    else:
        # Count only pixels on the target (e.g. primary target) object
        csum = np.sum(semantic_obs == sematic_id)
    per_on_obj = csum / res
    return per_on_obj


class SurfacePolicyCurvatureInformed(SurfacePolicy):
    """Policy class for a more intelligent surface-agent.

    Includes additional functions for moving along an object based on the direction
    of principle curvature (PC).

    A general summary of the policy is that the agent will start following PC directions
    as soon as these are well defined. This will initially be the minimal curvature; it
    will follow these for as long as they are defined, and until reaching a certain
    number of steps (max_pc_bias_steps), before then changing to follow maximal
    curvature. This process continues to alternate, as long as the PC directions are
    well defined.

    If PC are not meaningfully defined (which is often the case), then the agent uses
    standard momentum to take a step in a direction similar to the previous step
    (weighted by the alpha parameter); it will do this for a minimum number of steps
    (min_general_steps) before it will consider using PC information again.

    If the agent is taking a step that will bring it to a previously visited location
    according to its estimates, then it will attempt to correct for this and choose
    another direction; after finding a new heading, the agent performs a minimum number
    of steps (min_heading_steps) before it will check again for a conflicting heading.

    The main other event that can occur is that PCs are defined, but these are
    predominantly in the z-direction (relative to the agent); in that case, the
    PC-defined heading is ignored on that step, and a standard, momentum-based step is
    taken; such z-defined PC directions tend to occur on e.g. the rim of cups and
    are presumably due to issues with the agents re-orientation steps vs. noisiness
    of the PCs defined by the surface

    TODO update this method to accept more general notions of directions-of-variance
    (rather than strictly principal curvature), such that it can be applied in more
    abstract spaces
    """

    def __init__(
        self,
        alpha,
        pc_alpha,
        max_pc_bias_steps,
        min_general_steps,
        min_heading_steps,
        **kwargs,
    ):
        """Initialize policy.

        Args:
            alpha: to what degree should the move_tangentially direction be the
                same as the last step or totally random? 0~same as before, 1~random
                walk; used when not following principal curvature (PC)
            pc_alpha: the degree to which the moving average of the PC direction
                should be based on the history vs. the most recent step
            max_pc_bias_steps: the number of steps to take following a particular PC
                direction (i.e. maximum or minimal curvature) before switching to the
                other type of PC direction
            min_general_steps: the number of normal momentum steps that must be taken
                after the agent has stopped following principal curvature, and before it
                will follow PC again; if this is too low, then the agent has the habit
                of moving quite noisily as it keeps attempting to follow PC, but if it
                is too high, then PC directions are not used to their fullest
            min_heading_steps: when not following PC directions, the minimum number
                of steps to take before we check that we're not heading to a previously
                visited location; again, this should be sufficiently high that we don't
                bounce around noisily, but also low enough that we avoid revisiting
                locations
            **kwargs: Additional keyword arguments.
        """
        super().__init__(alpha, **kwargs)
        self.pc_alpha = pc_alpha

        # == Threshold variables that determine policy behaviour
        self.max_pc_bias_steps = max_pc_bias_steps
        self.min_general_steps = min_general_steps
        self.min_heading_steps = min_heading_steps

    def pre_episode(self):
        super().pre_episode()

        # == Variables for representing heading ==
        # We represent it both in angular and vector form as under different settings,
        # one or the other will be leveraged
        self.tangential_angle = None
        self.tangential_vec = None

        # == Boolean variables determining action decisions ==
        self.min_dir_pref = True  # Initialize direction preference for following the
        # minimal principal curvature direction

        # Token to indicate whether we have been following PC-guided
        # trajectories; useful for handling how we act when PC's are no longer
        # defined
        self.using_pc_guide = False

        # == Counter variables determining action decisions ==
        self.ignoring_pc_counter = self.min_general_steps  # How many steps we have
        # taken without PC-guidance; for the *first* time we encounter an informative
        # PC, we should ideally always consider it, therefore initialize to the minimum
        # required for beginning to follow PC
        self.following_heading_counter = 0  # Used to ensure that for normal tangential
        # steps, we follow a particular heading for a while before trying to select
        # a new one that avoids previous locations
        self.following_pc_counter = 0  # How long we've been following PC for; used to
        # keep track of when we should shift from following minimal to maximal curvature
        self.continuous_pc_steps = 0  # How many continuous steps in a row we have
        # taken for the PC (i.e. without entering a region with un-defined PCs); this
        # counter does *not* influence whether we follow minimal or maximal curvature;
        # instead, it helps us determine whether we have just got new PC information,
        # and therefore enables us the possibility of flipping the PC direction so as
        # to avoid previous locations

        # == Variables for estimating moving average of principal curvature direction ==
        self.prev_angle = None  # Used to accumulate previous directions
        # along a particular principle curvature, for e.g. moving average

        # == Variables to help us avoid revisiting previously observed locations ==
        self.tangent_locs = []  # Receive visited locations from sensor module,
        # specifically those associated with prev. tangential movements; helps us avoid
        # revisiting old locations; note we thus ignore locations associated with
        # the surface-agent-policy's re-orientation movements
        self.tangent_norms = []  # As for tangent_locs; helpful for distinguishing
        # locations as being on different surfaces

        # == Logging variables ==
        # Store detailed information about actions taken; useful for both visualization,
        # and potentially informing e.g. an LM to intervene where certain policy
        # decisions are failing
        if not hasattr(self, "action_details"):
            # TODO M clean up where we log action information for motor system
            # Ideally as much as as possible should be with buffer following refactor
            self.action_details = dict(
                pc_heading=[],  # "min", "max", or "no"; indicates the type of curvature
                # the agent is following
                avoidance_heading=[],  # True or False, whether the agent is taking a
                # new heading to avoid previously visited points
                z_defined_pc=[],  # None, and otherwise the principle curvature and
                # point normal directions when the PC direction is predominantly in
                # z-axis of the agent (i.e. pointing towards/away from the agent rather
                # than being orthogonal)
            )
        else:
            self.action_details.update(
                pc_heading=[],
                avoidance_heading=[],
                z_defined_pc=[],
            )

    def update_action_details(self):
        """Store informaton for later logging.

        This stores information that details elements of the policy or observations
        relevant to policy decisions.

        E.g. if model-free policy has been unable to find a path that avoids
        revisiting old locations, an LM might use this information to inform a
        particular action (TODO not yet implemented, and NOTE that any modelling
        should ultimately be located in the learning module(s), not in motor
        systems)
        """
        if self.using_pc_guide:
            if self.min_dir_pref:
                self.action_details["pc_heading"].append("min")
            else:
                self.action_details["pc_heading"].append("max")
        else:
            self.action_details["pc_heading"].append("no")

        self.action_details["avoidance_heading"].append(self.setting_new_heading)

        if self.pc_is_z_defined:
            # Note for logging we save the orientations in the global reference frame,
            # however whether the PC is z-defined is relative to the agent and its
            # orientation
            # TODO: This value doesn't seem to be used anywhere
            self.action_details["z_defined_pc"].append(
                [
                    self.processed_observations.get_point_normal(),
                    self.processed_observations.get_curvature_directions(),
                ]
            )
        else:
            self.action_details["z_defined_pc"].append(None)

    def tangential_direction(self) -> VectorXYZ:
        """Set the direction of action to be a direction 0 - 2pi.

        This controls the move_tangential action
            - start at 0 (go up in the reference frame of the agent, i.e. based on
            where it is facing), with the actual orientation determined via either
            principal curvature, or a random step weighted by momentum

        Tangential movements are the primary means of progressively exploring
        an object's surface

        Returns:
            VectorXYZ: direction of the action
        """
        # Reset booleans tracking z-axis PC directions and new headings
        self.pc_is_z_defined = False
        self.setting_new_heading = False

        logging.debug("Input-driven tangential movement")

        if (self.processed_observations.get_feature_by_name("pose_fully_defined")) and (
            self.ignoring_pc_counter >= self.min_general_steps
        ):  # Principal curvatures are defined, and counter for a min number of
            # general steps is satisfied

            tang_movement = self.perform_pc_guided_step()
            # Note this may fail if the PC guidance directs us back towards
            # a point we have previously experienced, in which case we revert
            # to a standard tangential movement (as below)

        # Use standard momentum (alpha weighting) if we've not found an area of
        # reliable principle curvature
        else:
            # Check whether we've just left a series of PC-defined trajectories
            # (i.e. entering a region of undefined PCs)
            if self.using_pc_guide:
                logging.debug("First movement after exiting PC-guided sequence")
                self.reset_pc_buffers()

            else:
                # Set PC-guidance flag; note the other elements of reset_pc_buffers
                # should/need not be reset
                self.using_pc_guide = False

            self.ignoring_pc_counter += 1

            tang_movement = self.perform_standard_tang_step()

        # Save detailed information about tangential steps
        self.update_action_details()

        return tang_movement

    def perform_pc_guided_step(self):
        """Inform steps to take using defined directions of principal curvature.

        Use the defined directions of principal curvature to inform (ideally a
        series) of steps along the appropriate direction.

        Returns:
            VectorXYZ: direction of the action
        """
        logging.debug("Attempting step with PC guidance")

        self.using_pc_guide = True

        self.check_for_preference_change()

        pc_for_use = self.determine_pc_for_use()  # Get the index for the PC we will
        # follow
        pc_dirs = self.processed_observations.get_curvature_directions()
        # Select the PC directions for use
        selected_pc_dir = pc_dirs[pc_for_use]

        # Rotate the tangential vector to be in the coordinate frame of the sensory
        # agent (rather than the global reference frame of the environment)
        inverse_quaternion_rotation = self.get_inverse_agent_rot()
        rotated_form = qt.rotate_vectors(inverse_quaternion_rotation, selected_pc_dir)

        # Before updating the representation and removing z-axis direction, check
        # for movements defined in the z-axis
        if int(np.argmax(np.abs(rotated_form))) == int(2):
            # TODO decide if in cases where the PC is defined in the z-direction
            # relative to the agent, it might actually make sense to still follow it,
            # i.e. as it should representa a movement tangential to the surface, and
            # suggests we're not oriented with the point-normal as expected; NB this
            # would need to be tested with the action-space of the surface-agent

            self.pc_is_z_defined = True

            logging.debug("Warning: PC is predominantly defined in z-direction")
            logging.debug("Skipping PC-guided move; using standard tangential movement")

            # Revert to a standard tangential step if we get to this stage
            # and the PC is predominantly defined in the z-direction; note that this
            # step will be weighted by the standard momentum, integrating the previous
            # movement, as we want to move in a consistent heading
            alternative_movement = self.perform_standard_tang_step()

            # Note that we *don't* re-set the PC buffers, because with any luck,
            # the PC axes will be better defined on the next step; it was also found in
            # practice that resetting the PC buffers (i.e. quitting entirely to
            # the PC-guided actions) occasionally resulted in longer, more noisy
            # inference

            return alternative_movement

        self.update_tangential_reps(vec_form=rotated_form)

        # Before attempting conflict avoidance, check if the PC direction itself
        # appears to have been arbitrarily flipped
        self.check_for_flipped_pc()

        # Check if new heading is necessary
        self.avoid_revisiting_locations()

        # If we are abandoning following PC directions, return the heading that was
        # found in the avoid_revisiting_locations search
        if self.setting_new_heading:
            # We've just found ourselves with a bad heading while following
            # principle curvatures, so break out of this; also reset the following
            # heading iter for standard tangential steps to make sure we move
            # sufficiently far from this area
            self.reset_pc_buffers()
            self.following_heading_counter = 0

            return qt.rotate_vectors(
                self.state["agent_id_0"]["rotation"],
                self.tangential_vec,
            )

        # Otherwise our heading is good; we continue and use our original heading (or
        # it's negative flip) to inform the PC heading
        logging.debug("We have a good PC heading")

        # Use a moving average of previous principal curvature directions to result in a
        # smoother path through areas where this shifts slightly
        if self.prev_angle is None:
            self.prev_angle = self.tangential_angle

        else:
            # Update tangential_vec using the moving average of the principle curvature
            self.pc_moving_average()

        self.following_pc_counter += 1
        self.continuous_pc_steps += 1

        return qt.rotate_vectors(
            self.state["agent_id_0"]["rotation"],
            self.tangential_vec,
        )

    def perform_standard_tang_step(self):
        """Perform a standard tangential step across the object.

        This is in contrast to, for example, being guided by principal curvatures.

        Note this is still more "intelligent" than the tangential step of the baseline
        surface-agent policy, because it also attempts to avoid revisiting old locations

        Returns:
            VectorXYZ: direction of the action
        """
        logging.debug("Standard tangential movement")

        # Select new movement, equivalent to alpha-weighted steps in the standard
        # informed surface-agent policy
        new_target_direction = (self.rng.rand() - 0.5) * 2 * np.pi

        if self.tangential_angle is not None:
            # Use alternative momentum calculation (vs. original SurfacePolicy
            # implementation)that does not suffer from discontinuous jump if new and
            # old theta are approximately -pi and +pi
            # NB the original surface-agent policy has not been updated as that method
            # appears to work as a "functional bug", where with an e.g. alpha=0.1, we
            # will *rarely* take a step in an unexpected direction, helping reduce the
            # chance of doing a continuous loop over an object with no exploration
            diff_amount = theta_change(self.tangential_angle, new_target_direction)
            blended_angle = self.tangential_angle + self.alpha * diff_amount

            blended_angle = enforce_pi_bounds(blended_angle)

            self.update_tangential_reps(angle_form=blended_angle)

        else:
            # On first movements, just use the initial direction
            self.update_tangential_reps(angle_form=new_target_direction)

        if self.following_heading_counter >= self.min_heading_steps:
            # Every now and then, check our heading is not in violation of prev. points,
            # and otherwise update it; thus, if
            # we need to avoid a certain direction, we will ignore momentum on this
            # particular iteration, continuing it on the next step
            self.avoid_revisiting_locations()
            # Note the value for self.tangential_vec and self.tangential_angle is
            # updated by avoid_revisiting_locations (if necessary)

            if self.setting_new_heading:
                logging.debug("Ignoring momentum to avoid prev. locations")

                self.following_heading_counter = (
                    0  # Reset this counter, so that we continue on the new heading
                )

            elif not self.setting_new_heading:
                # Either due to the original heading being fine, or due to timing out
                # in the search, we continue with our original heading
                pass

        self.following_heading_counter += 1

        return qt.rotate_vectors(
            self.state["agent_id_0"]["rotation"],
            self.tangential_vec,
        )

    def update_tangential_reps(self, vec_form=None, angle_form=None):
        """Update the angle and vector representation of a tangential heading.

        Angle and vector representations are stored as self.tangential_angle and
        self.tangential_vec, respectively.

        Ensures the two representations are always consistent. Further ensures that,
        because these movements are tangential, it will be defined in the plane,
        relative to the agent (i.e. movement along x and y only), and any inadvertant
        movement along the z-axis relative to the agent will be eliminated. User should
        supply *either* the vector form or the (Euler) angle form in radians that will
        define the new representations.
        """
        logging.debug("Updating tangential representations")
        assert (vec_form is None) != (
            angle_form is None  # Logical xor
        ), "Please provide one format for updating the tangential representation"
        if vec_form is not None:
            self.tangential_angle = projected_angle_from_vec(vec_form)
            self.tangential_vec = projected_vec_from_angle(self.tangential_angle)
        elif angle_form is not None:
            self.tangential_vec = projected_vec_from_angle(angle_form)
            self.tangential_angle = projected_angle_from_vec(self.tangential_vec)
        logging.debug(f"Angular form {self.tangential_angle}; vector form:")
        logging.debug(self.tangential_vec)

    def reset_pc_buffers(self):
        """Reset counters and other variables.

        We've just left a series of PC-defined trajectories (i.e. entered a region
        of undefined PCs), or had to select a new heading in order to avoid revisiting
        old locations. As such, appropriately reset counters and other variables.

        Note we do not reset tangential_angle or the vector, such that this information
        can still be used by e.g. momentum on the next step to keep us going generally
        forward
        """
        self.ignoring_pc_counter = 0  # Reset counter as we've just started
        # ignoring the PC after a period of movements

        self.using_pc_guide = False  # We're not using PC directions

        self.continuous_pc_steps = 0  # Note we don't reset self.following_pc_counter
        # because we want to keep our bias of PC heading for when we re-enter
        # a region where PC is defined again

        # Reset variables used for determining moving average estimate of the PC
        # direction
        self.prev_angle = None

    def check_for_preference_change(self):
        """Flip the preference for the min or max PC after a certain number of steps.

        This way, we can more quickly explore different "parts" of an object, rather
        than just persistently following e.g. the rim of a cup. By default, there is
        always an initial bias for the smallest principal curvature.

        TODO can eventually combine with a hypothesis-testing policy that encourages
        exploration of unvisited parts of the object, rather than relying on this
        simple counter-heuristic
        """
        if self.following_pc_counter == self.max_pc_bias_steps:
            logging.debug("Changing preference for pc-type.")
            logging.debug(f"Previous pref. for min-directions: {self.min_dir_pref}")

            self.min_dir_pref = not (self.min_dir_pref)

            self.following_pc_counter = 0  # Reset to allow multiple steps in
            # new chosen direction
            self.continuous_pc_steps = 0  # Reset so that we can flip the new PC
            # direction if necessary (i.e. if there is a risk of revisiting old
            # locations)

            # Reset moving average of principle curvature directions, as this is going
            # to be orthogonal to previous estimates
            self.prev_angle = None

            logging.debug(f"Updated preference: {self.min_dir_pref}")

    def determine_pc_for_use(self):
        """Determine the principal curvature to use for our heading.

        Use magnitude (ignoring negatives), as well as the current direction
        preference.

        Returns:
            Principal curvature to use.
        """
        absolute_pcs = np.abs(
            self.processed_observations.get_feature_by_name("principal_curvatures")
        )

        if self.min_dir_pref:  # Follow minimal curvature direction
            return np.argmin(absolute_pcs)

        else:
            return np.argmax(absolute_pcs)

    def avoid_revisiting_locations(
        self,
        conflict_divisor=3,
        max_steps=100,
    ):
        """Avoid revisiting locations.

        Check if the new proposed location direction is already pointing to somewhere
        we've visited before; if not, we can use the initially proposed movement.

        If there is a conflict, we select a new heading that avoids this; this is
        achieved by iteratively searching for a heading that does not conflict with any
        previously visited locations.

        Args:
            conflict_divisor: The amount pi is divided by to determine that a current
                heading will be too close to a previously visited location; this is an
                initial value that will be dynamically adjusted. Defaults to 3.
            max_steps: Maximum iterations of the search to perform to try to find a
                non-conflicting heading. Defaults to 100.

        Note that while the policy might have "unrealistic" access to information about
        it's location in the environment, this could easily be replaced by relative
        locations based on the first sensation

        Finally, note that in many situations, revisiting locations can be a good thing
        (e.g. re-anchoring given noisy path-integration), so we may want to activate
        /inactivate this as necessary (TODO)

        TODO separate out avoid_revisiting_locations as its own mixin so that it can
        be used more broadly
        """
        self.conflict_divisor = conflict_divisor
        self.max_steps = max_steps

        # Backup the original tangential vector
        vec_copy = copy.copy(self.tangential_vec)

        if len(self.tangent_locs) > 0:  # Only relevant if prev. locations visited
            inverse_quaternion_rotation = self.get_inverse_agent_rot()

            current_loc = self.tangent_locs[-1]
            logging.debug("Checking we don't head for prev. locations")
            logging.debug(f"Current location: {current_loc}")

            # Previous locations relative to current one (i.e. centered)
            adjusted_prev_locs = np.array(self.tangent_locs) - current_loc

            # Further adjust the relative locations based on the reference frame of the
            # moving SM-agent; we are going to look for conflicts by comparing these
            # locations to the headings (also in the reference frame of the agent) that
            # we might take
            # TODO could vectorize this
            rotated_locs = qt.rotate_vectors(
                inverse_quaternion_rotation, adjusted_prev_locs
            )

            # Until we have not found a direction that we can guarentee is
            # in a new heading, continue to attempt new directions
            searching_for_heading = True

            self.first_attempt = True  # On the first attempt to fix the direction
            # heading, simply flip it's orientation; this will often resolve issues
            # with principal curvature spontaneously swapping direction

            self.setting_new_heading = False  # If the heading isn't already good, or
            # *(-1) to fix an arbitrarily flipped PC direction is not sufficient,
            # then set this to True and abandon following PCs, reverting to a standard
            # momentum operation

            self.search_counter = 0  # Eventually, abort the search if we exceed a
            # threshold; additionally, as this increases, we periodically decrease the
            # angle we are concerned with for conflicts, making it more likely that
            # we at least avoid being nearby to previous points

            while searching_for_heading:
                conflicts = False  # Assume False until evidence otherwise

                logging.debug(f"Number of locations to check : {len(rotated_locs) - 1}")

                # Check all prev. locations for conflict; TODO could vectorize this
                for ii in range(
                    max(0, len(rotated_locs) - 50), len(rotated_locs) - 1
                ):  # Ignore current point, and points before the last 50 sensations
                    # Check if there is actually a conflict
                    on_conflict = self.conflict_check(rotated_locs, ii)

                    if on_conflict:
                        conflicts = True  # Keep track of the fact that we've found
                        # a conflicting direction that we need to deal with

                        logging.debug("Angle is low, so re-orienting")
                        logging.debug(f"Inducing location from sensation {ii}")
                        logging.debug(f"Currently on sensation {len(rotated_locs)}")

                        self.attempt_conflict_resolution(vec_copy)

                if not conflicts:  # We have a valid heading
                    searching_for_heading = False

                    logging.debug("The final direction from conflict checking:")
                    logging.debug(self.tangential_vec)
                    self.update_tangential_reps(vec_form=self.tangential_vec)

                elif self.search_counter >= self.max_steps:
                    logging.debug("Abandoning search, no directions without conflict")
                    logging.debug("Therefore using original headng")

                    self.update_tangential_reps(vec_form=vec_copy)

                    return None

                else:
                    # Search continues, but occasionally narrow the region in which we
                    # look for conflicts, making it easier to select a path "out"
                    self.search_counter += 1
                    if self.search_counter % 20 == 0:
                        self.conflict_divisor += 1

                        logging.debug(
                            f"Updating conflict divisor: {self.conflict_divisor}"
                        )

    def conflict_check(self, rotated_locs, ii):
        """Check for conflict in the current heading.

        Target location needs to be similar *and* we need to have a similar point normal
        to discount the current proposed heading; if point normals are significantly
        different, then we are likely on a different surface, in which case passing by
        nearby previous points is no longer problematic.

        Note that when executing failed hypothesis-testing jumps, we can have multiple
        instances of the same location in our history; this will result in
        get_angle_beefed_up returning infinity, i.e. we don't worry about avoiding our
        current location

        Returns:
            True if there is a conflict, False otherwise.
        """
        assert np.linalg.norm(rotated_locs[-1]) == 0, "Should be centered to 0"

        if (
            (
                get_angle_beefed_up(self.tangential_vec, rotated_locs[ii])
                <= np.pi / self.conflict_divisor
            )
            and (
                # Heuristic of np.pi / 4 for point-normals is that sides of
                # a cube will therefore be different surfaces, but not necessarily
                # points along e.g. a bowl's curved underside
                get_angle_beefed_up(self.tangent_norms[-1], self.tangent_norms[ii])
                <= np.pi / 4
            )
            and (
                # Only consider points that are relatively close by (2.5 cm) in
                # Euclidean space; note we are comparing to an origin of 0,0,0
                # Re. choice of 2.5 cm, tangential steps are generally of size 0.004
                # (4mm), so this seems like a reasonable estimate
                np.linalg.norm(rotated_locs[ii], ord=2) <= 0.025
            )
        ):
            return True
        else:
            return False

    def attempt_conflict_resolution(self, vec_copy):
        """Try to define direction vector that avoids revisiting previous locations."""
        if self.first_attempt and self.using_pc_guide and self.continuous_pc_steps == 0:
            # On the first PC-guided step in a series, try to choose a direction
            # along the PC that is away from previous locations
            # Note first-attempt refers to trying to fix a bad PC
            # heading with a flip first; otherwise we have to abort
            # following PC
            # Note that when not concerned with principal curvatures,
            # we do not try this initial flip

            self.update_tangential_reps(vec_form=np.array(vec_copy) * -1)
            logging.debug("Flipping the PC direction to avoid prev. locations.")
            self.first_attempt = False

        else:
            logging.debug("Trying a new random heading")
            self.setting_new_heading = True  # PC, if it was being followed, will
            # be abandoned

            # Progressively rotate in the plane, depending on how
            # long we've been searching, and starting with a small
            # possible rotation
            rot_limits = np.clip(0.1 + self.search_counter / self.max_steps, 0, 1)
            rot_val = self.rng.uniform(-rot_limits, rot_limits) * np.pi

            plane_rot = rot.from_euler("xyz", (0.0, 0.0, rot_val), degrees=False)

            # Note we apply the rotation to the original vector
            self.update_tangential_reps(vec_form=plane_rot.apply(vec_copy))

    def check_for_flipped_pc(self):
        """Check for arbitrarily flipped PC direction.

        Do a quick check to see if the previous PC heading has been arbitrarily
        flipped, in which case, flip it back. With any luck, this will allow us to
        automatically pass avoid_revisiting_locations and thereby continue using PCs
        where relevant.
        """
        if self.prev_angle is not None:
            if (
                abs(theta_change(self.prev_angle, self.tangential_angle + np.pi))
                <= np.pi / 4
            ):
                logging.debug(
                    "Evidence that PC direction arbitrarily flipped, flipping back"
                )
                self.update_tangential_reps(vec_form=np.array(self.tangential_vec) * -1)

    def pc_moving_average(self):
        """Calculate a moving average of the principal curvature direction.

        The moving average should be consistent even on curved surfaces as the
        directions will be in the reference frame of the agent (which has rotated itself
        to align with the point normal)
        """
        logging.debug("Applying momentum to PC direction estimate")

        # Calculate a weighted average in such a way that ensures we can handle
        # the circular nature of angles (i.e. that +pi and -pi are adjacent)
        diff_amount = theta_change(self.prev_angle, self.tangential_angle)
        blended_angle = self.tangential_angle + self.pc_alpha * diff_amount

        blended_angle = enforce_pi_bounds(blended_angle)

        self.update_tangential_reps(angle_form=blended_angle)

        # Store action taken for estimating future movements
        self.prev_angle = self.tangential_angle


def theta_change(a, b):
    """Determine the min, signed change in orientation between two angles in radians.

    Returns:
        Signed change in orientation.
    """
    min_sep = a - b

    # If an angle has "looped" round, correct for this
    min_sep = enforce_pi_bounds(min_sep)

    return min_sep


def enforce_pi_bounds(theta):
    """Enforce an orientation to be bounded between - pi and + pi.

    Returns:
        Angle in radians.
    """
    while theta > np.pi:
        theta -= np.pi * 2
    while theta < -np.pi:
        theta += np.pi * 2

    return theta


def projected_angle_from_vec(vector):
    """Determine the rotation about a z-axis (pointing "up").

    Note that because of the convention for moving along the y when theta=0, the
    typical order of arguments is swapped from calculating the standard
    atan2 (https://en.wikipedia.org/wiki/Atan2).

    Returns:
        Angle in radians.
    """
    test_thetas = [-np.pi, -np.pi / 2, -0.1, 0, 0.1, np.pi / 2, np.pi]

    # Double check that we correctly recover the theta over possible range
    for test_theta in test_thetas:
        test_vec = projected_vec_from_angle(test_theta)
        recovered = math.atan2(test_vec[0], test_vec[1])
        assert (
            abs(test_theta - recovered) < 0.01
        ), f"Issue with angle recovery for : {test_theta} vs. {recovered}"

    return math.atan2(vector[0], vector[1])


def projected_vec_from_angle(angle):
    """Determine the vector in the plane defined by an orientation around the z-axis.

    Takes angle in radians, bound between -pi : pi.

    This continues the convention started in the original surface-agent policy that a
    theta of 0 should correspond to a movement of 1 in the y direction, and 0 in the x
    direction, rather than vice-versa; this convention is implemented by the np.pi/2
    offsets.

    Returns:
        Vector in the plane defined by an orientation around the z-axis.
    """
    assert abs(angle) < np.pi + 0.01, f"-pi : +pi bound angles only : {angle}"

    return [np.cos(angle - np.pi / 2), np.sin(angle + np.pi / 2), 0]
