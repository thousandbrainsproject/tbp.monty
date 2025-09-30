# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import copy
import logging
from pprint import pformat

import numpy as np
import quaternion
from typing_extensions import Self

from tbp.monty.frameworks.actions.action_samplers import UniformlyDistributedSampler
from tbp.monty.frameworks.actions.actions import (
    Action,
    LookUp,
    MoveTangentially,
    OrientVertical,
    SetAgentPose,
    SetSensorRotation,
)
from tbp.monty.frameworks.models.motor_policies import (
    GetGoodView,
    InformedPolicy,
    ObjectNotVisible,
    PositioningProcedure,
    SurfacePolicy,
)
from tbp.monty.frameworks.models.motor_system import MotorSystem
from tbp.monty.frameworks.models.motor_system_state import (
    MotorSystemState,
    ProprioceptiveState,
)

from .embodied_environment import EmbodiedEnvironment

__all__ = [
    "EnvironmentDataLoader",
    "EnvironmentDataLoaderPerObject",
    "InformedEnvironmentDataLoader",
    "OmniglotDataLoader",
    "SaccadeOnImageDataLoader",
    "SaccadeOnImageFromStreamDataLoader",
]

logger = logging.getLogger(__name__)


class EnvironmentDataLoader:
    """Provides an interface to an embodied environment.

    The observations are based on the actions returned by the `motor_system`.

    The first values returned by this iterator are the observations of the
    environment's initial state, subsequent observations are returned after the action
    returned by `motor_system` is applied.

    Attributes:
        env: An instance of a class that implements :class:`EmbodiedEnvironment`.
        motor_system: :class:`MotorSystem`
        rng: Random number generator to use.
        transform: Callable used to transform the observations returned by
            the environment.

    Note:
        If the amount variable returned by motor_system is None, the amount used by
        habitat will be the default for the actuator, e.g.
        PanTiltZoomCamera.translation_step

    Note:
        This one on its own won't work.

    Raises:
        TypeError: If `motor_system` is not an instance of `MotorSystem`.
    """

    def __init__(
        self, env: EmbodiedEnvironment, motor_system: MotorSystem, rng, transform=None
    ):
        if not isinstance(motor_system, MotorSystem):
            raise TypeError(
                f"motor_system must be an instance of MotorSystem, got {motor_system}"
            )
        self.env = env
        self.motor_system = motor_system
        self.rng = rng
        self.transform = transform
        if self.transform is not None:
            for t in self.transform:
                if t.needs_rng:
                    t.rng = self.rng
        self._observation, proprioceptive_state = self.reset()
        self.motor_system._state = (
            MotorSystemState(proprioceptive_state) if proprioceptive_state else None
        )
        self._action = None
        self._counter = 0

    @property
    def action_space(self):
        return self.env.action_space

    def __iter__(self) -> Self:
        """Implement the iterator protocol.

        Returns:
            The iterator.
        """
        return self

    def __next__(self):
        if self._counter == 0:
            # Return first observation after 'reset' before any action is applied
            self._counter += 1
            return self._observation
        else:
            action = self.motor_system()
            self._action = action
            self._observation, proprioceptive_state = self.step(action)
            self.motor_system._state = (
                MotorSystemState(proprioceptive_state) if proprioceptive_state else None
            )
            self._counter += 1
            return self._observation

    def reset(self):
        observation = self.env.reset()
        state = self.env.get_state()

        if self.transform is not None:
            observation = self.apply_transform(self.transform, observation, state)
        return observation, ProprioceptiveState(state) if state else None

    def apply_transform(self, transform, observation, state):
        if isinstance(transform, list):
            for t in transform:
                observation = t(observation, state)
        else:
            observation = transform(observation, state)
        return observation

    def step(self, action: Action):
        observation = self.env.step(action)
        state = self.env.get_state()
        if self.transform is not None:
            observation = self.apply_transform(self.transform, observation, state)
        return observation, ProprioceptiveState(state) if state else None

    def pre_episode(self):
        self.motor_system.pre_episode()

        # Reset the data loader state.
        self._observation, proprioceptive_state = self.reset()
        self.motor_system._state = (
            MotorSystemState(proprioceptive_state) if proprioceptive_state else None
        )
        self._action = None
        self._counter = 0

    def post_episode(self):
        self.motor_system.post_episode()

    def pre_epoch(self):
        pass

    def post_epoch(self):
        pass


class EnvironmentDataLoaderPerObject(EnvironmentDataLoader):
    """Dataloader for testing on environment with one "primary target" object.

    Dataloader for testing on environment where we load one "primary target" object
    at a time; in addition, one can optionally load other "distractor" objects to the
    environment

    Has a list of primary target objects, swapping these objects in and out for episodes
    without resetting the environment. The objects are initialized with parameters such
    that we can vary their location, rotation, and scale.

    After the primary target is added to the environment, other distractor objects,
    sampled from the same object list, can be added.
    """

    def __init__(self, object_names, object_init_sampler, *args, **kwargs):
        """Initialize dataloader.

        Args:
            object_names: list of objects if doing a simple experiment with primary
                target objects only; dict for experiments with multiple objects,
                corresponding to -->
                targets_list : the list of primary target objects
                source_object_list : the original object list from which the primary
                    target objects were sampled; used to sample distractor objects
                num_distractors : the number of distractor objects to add to the
                    environment
            object_init_sampler: Function that returns dict with position, rotation,
                and scale of objects when re-initializing. To keep configs
                serializable, default is set to :class:`DefaultObjectInitializer`.
            *args: ?
            **kwargs: ?

        See Also:
            tbp.monty.frameworks.make_dataset_configs
            :class:`EnvironmentDataLoaderPerObjectTrainArgs`

        Raises:
            TypeError: If `object_names` is not a list or dictionary
        """
        super(EnvironmentDataLoaderPerObject, self).__init__(*args, **kwargs)
        if isinstance(object_names, list):
            self.object_names = object_names
            # Return an (ordered) list of unique items:
            self.source_object_list = list(dict.fromkeys(object_names))
            self.num_distractors = 0
        elif isinstance(object_names, dict):
            # TODO when we want more advanced multi-object experiments, update these
            # arguments along with the Object Initializers so that we can easily
            # specify a set of primary targets and distractors, i.e. random sampling
            # of the distractor objects shouldn't happen here
            self.object_names = object_names["targets_list"]
            self.source_object_list = list(
                dict.fromkeys(object_names["source_object_list"])
            )
            self.num_distractors = object_names["num_distractors"]
        else:
            raise TypeError("Object names should be a list or dictionary")
        self.create_semantic_mapping()

        self.object_init_sampler = object_init_sampler
        self.object_init_sampler.rng = self.rng
        self.object_params = self.object_init_sampler()
        self.current_object = 0
        self.n_objects = len(self.object_names)
        self.episodes = 0
        self.epochs = 0
        self.primary_target = None

    def pre_episode(self):
        super().pre_episode()

        self.motor_system._state[self.motor_system._policy.agent_id][
            "motor_only_step"
        ] = False

    def post_episode(self):
        super().post_episode()
        self.object_init_sampler.post_episode()
        self.object_params = self.object_init_sampler()
        self.cycle_object()
        self.episodes += 1

    def pre_epoch(self):
        self.change_object_by_idx(0)

    def post_epoch(self):
        self.epochs += 1
        self.object_init_sampler.post_epoch()
        self.object_params = self.object_init_sampler()

    def create_semantic_mapping(self):
        """Create a unique semantic ID (positive integer) for each object.

        Used by Habitat for the semantic sensor.

        In addition, create a dictionary mapping back and forth between these IDs and
        the corresponding name of the object
        """
        assert set(self.object_names).issubset(set(self.source_object_list)), (
            "Semantic mapping requires primary targets sampled from source list"
        )

        starting_integer = 1  # Start at 1 so that we can distinguish on-object semantic
        # IDs (>0) from being off object (semantic_id == 0 in Habitat by default)
        self.semantic_id_to_label = {
            i + starting_integer: label
            for i, label in enumerate(self.source_object_list)
        }
        self.semantic_label_to_id = {
            label: i + starting_integer
            for i, label in enumerate(self.source_object_list)
        }

    def cycle_object(self):
        """Remove the previous object(s) from the scene and add a new primary target.

        Also add any potential distractor objects.
        """
        next_object = (self.current_object + 1) % self.n_objects
        logger.info(
            f"\n\nGoing from {self.current_object} to {next_object} of {self.n_objects}"
        )
        self.change_object_by_idx(next_object)

    def change_object_by_idx(self, idx):
        """Update the primary target object in the scene based on the given index.

        The given `idx` is the index of the object in the `self.object_names` list,
        which should correspond to the index of the object in the `self.object_params`
        list.

        Also add any distractor objects if required.

        Args:
            idx: Index of the new object and ints parameters in object_params
        """
        assert idx <= self.n_objects, "idx must be <= self.n_objects"
        self.env.remove_all_objects()

        # Specify config for the primary target object and then add it
        init_params = self.object_params.copy()
        init_params.pop("euler_rotation")
        if "quat_rotation" in init_params.keys():
            init_params.pop("quat_rotation")
        init_params["semantic_id"] = self.semantic_label_to_id[self.object_names[idx]]

        # TODO clean this up with its own specific call i.e. Law of Demeter
        primary_target_obj = self.env.add_object(
            name=self.object_names[idx], **init_params
        )

        if self.num_distractors > 0:
            self.add_distractor_objects(
                primary_target_obj,
                init_params,
                primary_target_name=self.object_names[idx],
            )

        self.current_object = idx
        self.primary_target = {
            "object": self.object_names[idx],
            "semantic_id": self.semantic_label_to_id[self.object_names[idx]],
            **self.object_params,
        }
        logger.info(f"New primary target: {pformat(self.primary_target)}")

    def add_distractor_objects(
        self, primary_target_obj, init_params, primary_target_name
    ):
        """Add arbitrarily many "distractor" objects to the environment.

        Args:
            primary_target_obj : the Habitat object which is the primary target in
                the scene
            init_params: parameters used to initialize the object, e.g.
                orientation; for now, these are identical to the primary target
                except for the object ID
            primary_target_name: name of the primary target object
        """
        # Sample distractor objects from those that are not the primary target; this
        # is so that, for now, we can evaluate how well the model stays on the primary
        # target object until it is classified, with no ambiguity about what final
        # object it is classifying
        sampling_list = [
            item for item in self.source_object_list if item != primary_target_name
        ]

        for __ in range(self.num_distractors):
            new_init_params = copy.deepcopy(init_params)

            new_obj_label = self.rng.choice(sampling_list)
            new_init_params["semantic_id"] = self.semantic_label_to_id[new_obj_label]
            # TODO clean up the **unpacking used
            self.env.add_object(
                name=new_obj_label,
                **new_init_params,
                object_to_avoid=True,
                primary_target_object=primary_target_obj,
            )


class InformedEnvironmentDataLoader(EnvironmentDataLoaderPerObject):
    """Dataloader that supports a policy which makes use of previous observation(s).

    Extension of the EnvironmentDataLoader where the actions can be informed by the
    observations. It passes the observation to the InformedPolicy class (which is an
    extension of the BasePolicy). This policy can then make use of the observation
    to decide on the next action.

    Also has the following, additional functionality; TODO refactor/separate these
    out as appropriate

    i) this dataloader allows for early stopping by adding the set_done
    method which can for example be called when the object is recognized.

    ii) the motor_only_step can be set such that the sensory module can
    later determine whether perceptual data should be sent to the learning module,
    or just fed back to the motor policy.

    iii) Handles different data-loader updates depending on whether the policy is
    based on the surface-agent or touch-agent

    iv) Supports hypothesis-testing "jump" policy
    """

    def __next__(self):
        if self._counter == 0:
            return self.first_step()

        # Check if any LM's have output a goal-state (such as hypothesis-testing
        # goal-state)
        elif (
            isinstance(self.motor_system._policy, InformedPolicy)
            and self.motor_system._policy.use_goal_state_driven_actions
            and self.motor_system._policy.driving_goal_state is not None
        ):
            return self.execute_jump_attempt()

        # NOTE: terminal conditions are now handled in experiment.run_episode loop
        else:
            attempting_to_find_object = False
            try:
                self._action = self.motor_system()
            except ObjectNotVisible:
                # Note: Only SurfacePolicy raises ObjectNotVisible.
                attempting_to_find_object = True
                self._action = self.motor_system._policy.touch_object(
                    self._observation,
                    view_sensor_id="view_finder",
                    state=self.motor_system._state,
                )
            else:
                # TODO: Encapsulate this reset inside TouchObject positioning
                #       procedure once it exists.
                #       This is a hack to reset the current touch_object
                #       positioning procedure state so that the next time
                #       SurfacePolicy falls off the object, it will try to find
                #       the object using its full repertoire of actions.
                self.motor_system._policy.touch_search_amount = 0

            self._observation, proprioceptive_state = self.step(self._action)
            motor_system_state = MotorSystemState(proprioceptive_state)

            # TODO: Refactor this so that all of this is contained within the
            #       SurfacePolicy and/or positioning procedure.
            if isinstance(self.motor_system._policy, SurfacePolicy):
                # When we are attempting to find the object, we are always performing
                # a motor-only step.
                motor_system_state[self.motor_system._policy.agent_id][
                    "motor_only_step"
                ] = attempting_to_find_object

                if (
                    not attempting_to_find_object
                    and self._action.name != OrientVertical.action_name()
                ):
                    # We are not attempting to find the object, which means that we
                    # are executing the SurfacePolicy.dynamic_call action cycle.
                    # Out of the four actions in the
                    # MoveForward->OrientHorizontal->OrientVertical->MoveTangentially
                    # "subroutine" defined in SurfacePolicy.dynamic_call, we only
                    # want to send data to the learning module after taking the
                    # OrientVertical action. The other three actions in the cycle
                    # are motor-only to keep the surface agent on the object.
                    motor_system_state[self.motor_system._policy.agent_id][
                        "motor_only_step"
                    ] = True

            self.motor_system._state = motor_system_state

            if not attempting_to_find_object:
                self._counter += 1

            return self._observation

    def pre_episode(self):
        super().pre_episode()
        if self.env._agents[0].action_space_type != "surface_agent":
            on_target_object = self.get_good_view_with_patch_refinement()
            if self.num_distractors == 0:
                # Only perform this check if we aren't doing multi-object experiments.
                assert on_target_object, (
                    "Primary target must be visible at the start of the episode"
                )

    def first_step(self):
        """Carry out particular motor-system state updates required on the first step.

        TODO ?can get rid of this by appropriately initializing motor_only_step

        Returns:
            The observation from the first step.
        """
        # Return first observation after 'reset' before any action is applied
        self._counter += 1

        # Based on current code-base self._action will always be None when
        # the counter is 0
        assert self._action is None, "Setting of motor_only_step may need updating"

        # For first step of surface-agent policy, always bypass LM processing
        # For distant-agent policy, we still process the first sensation if it is
        # on the object
        self.motor_system._state[self.motor_system._policy.agent_id][
            "motor_only_step"
        ] = isinstance(self.motor_system._policy, SurfacePolicy)

        return self._observation

    def get_good_view(
        self,
        sensor_id: str,
        allow_translation: bool = True,
        max_orientation_attempts: int = 1,
    ) -> bool:
        """Invoke the GetGoodView positioning procedure.

        Args:
            sensor_id: The ID of the sensor to use for positioning.
            allow_translation: Whether to allow movement toward the object via
                the motor systems's move_close_enough method. If False, only
                orientienting movements are performed. Defaults to True.
            max_orientation_attempts: The maximum number of orientation attempts
                allowed before giving up and truncating the procedure indicating that
                the sensor is not on the target object.

        Returns:
            Whether the sensor is on the target object.
        """
        positioning_procedure = GetGoodView(
            agent_id=self.motor_system._policy.agent_id,
            desired_object_distance=self.motor_system._policy.desired_object_distance,
            good_view_percentage=self.motor_system._policy.good_view_percentage,
            multiple_objects_present=self.num_distractors > 0,
            sensor_id=sensor_id,
            target_semantic_id=self.primary_target["semantic_id"],
            allow_translation=allow_translation,
            max_orientation_attempts=max_orientation_attempts,
            # TODO: Remaining arguments are unused but required by BasePolicy.
            #       These will be removed when PositioningProcedure is split from
            #       BasePolicy
            #
            # Note that if we use rng=self.rng below, then the following test will
            # fail:
            #   tests/unit/evidence_lm_test.py::EvidenceLMTest::test_two_lm_heterarchy_experiment  # noqa: E501
            # The test result seems to be coupled to the random seed and the
            # specific sequence of rng calls (rng is called once on GetGoodView
            # initialization).
            rng=np.random.RandomState(),
            action_sampler_args=dict(actions=[LookUp]),
            action_sampler_class=UniformlyDistributedSampler,
            switch_frequency=0.0,
        )
        result = positioning_procedure.positioning_call(
            self._observation, self.motor_system._state
        )
        while not result.terminated and not result.truncated:
            for action in result.actions:
                self._observation, proprio_state = self.step(action)
                self.motor_system._state = (
                    MotorSystemState(proprio_state) if proprio_state else None
                )

            result = positioning_procedure.positioning_call(
                self._observation, self.motor_system._state
            )

        return result.success

    def get_good_view_with_patch_refinement(self) -> bool:
        """Policy to get a good view of the object for the central patch.

        Used by the distant agent to move and orient toward an object such that the
        central patch is on-object. This is done by first moving and orienting the
        agent toward the object using the view finder. Then orienting movements are
        performed using the central patch (i.e., the sensor module with id
        "patch" or "patch_0") to ensure that the patch's central pixel is on-object.
        Up to 3 reorientation attempts are performed using the central patch.

        Also currently used by the distant agent after a "jump" has been initialized
        by a model-based policy.

        Returns:
            Whether the sensor is on the object.

        """
        self.get_good_view("view_finder")
        for patch_id in ("patch", "patch_0"):
            if patch_id in self._observation["agent_id_0"].keys():
                on_target_object = self.get_good_view(
                    patch_id,
                    allow_translation=False,  # only orientation movements
                    max_orientation_attempts=3,  # allow 3 reorientation attempts
                )
                break
        return on_target_object

    def execute_jump_attempt(self):
        """Attempt a hypothesis-testing "jump" onto a location of the object.

        Delegates to motor policy directly to determine specific jump actions.

        Returns:
            The observation from the jump attempt.
        """
        logger.debug(
            "Attempting a 'jump' like movement to evaluate an object hypothesis"
        )

        # Store the current location and orientation of the agent
        # If the hypothesis-guided jump is unsuccesful (e.g. to empty space,
        # or inside an object, we return here)
        pre_jump_state = self.motor_system._state[self.motor_system._policy.agent_id]

        # Check that all sensors have identical rotations - this is because actions
        # currently update them all together; if this changes, the code needs
        # to be updated; TODO make this its own method
        for ii, current_sensor in enumerate(pre_jump_state["sensors"].keys()):
            if ii == 0:
                first_sensor = current_sensor
            assert np.all(
                pre_jump_state["sensors"][current_sensor]["rotation"]
                == pre_jump_state["sensors"][first_sensor]["rotation"]
            ), "Sensors are not identical in pose"

        # TODO In general what would be best/cleanest way of routing information,
        # e.g. perhaps the learning module should just pass a *displacement* (in
        # internal coordinates, and a target surface normal)
        # Could also consider making use of decide_location_for_movement (or
        # decide_location_for_movement_matching)

        (target_loc, target_np_quat) = (
            self.motor_system._policy.derive_habitat_goal_state()
        )

        # Update observations and motor system-state based on new pose, accounting
        # for resetting both the agent, as well as the poses of its coupled sensors;
        # this is necessary for the distant agent, which pivots the camera around
        # like a ball-and-socket joint; note the surface agent does not
        # modify this from the the unit quaternion and [0, 0, 0] position
        # anyways; further note this is globally applied to all sensors.
        set_agent_pose = SetAgentPose(
            agent_id=self.motor_system._policy.agent_id,
            location=target_loc,
            rotation_quat=target_np_quat,
        )
        set_sensor_rotation = SetSensorRotation(
            agent_id=self.motor_system._policy.agent_id,
            rotation_quat=quaternion.one,
        )
        _, _ = self.step(set_agent_pose)
        self._observation, proprioceptive_state = self.step(set_sensor_rotation)
        self.motor_system._state = (
            MotorSystemState(proprioceptive_state) if proprioceptive_state else None
        )

        # Check depth-at-center to see if the object is in front of us
        # As for methods such as touch_object, we use the view-finder
        depth_at_center = PositioningProcedure.depth_at_center(
            agent_id=self.motor_system._policy.agent_id,
            observation=self._observation,
            sensor_id="view_finder",
        )

        # If depth_at_center < 1.0, there is a visible element within 1 meter of the
        # view-finder's central pixel)
        if depth_at_center < 1.0:
            self.handle_successful_jump()

        else:
            self.handle_failed_jump(pre_jump_state, first_sensor)

        # Regardless of whether movement was successful, counts as a step,
        # and we provide the observation to the next step of the motor policy
        self._counter += 1

        self.motor_system._state[self.motor_system._policy.agent_id][
            "motor_only_step"
        ] = True

        # TODO refactor so that the whole of the hypothesis driven jumps
        # makes cleaner use of self.motor_system()
        # Call post_action (normally taken care of __call__ within
        # self.motor_system._policy())
        self.motor_system._policy.post_action(
            self.motor_system._policy.action, self.motor_system._state
        )

        return self._observation

    def handle_successful_jump(self):
        """Deal with the results of a successful hypothesis-testing jump.

        A successful jump is "on-object", i.e. the object is perceived by the sensor.
        """
        logger.debug(
            "Object visible, maintaining new pose for hypothesis-testing action"
        )

        if isinstance(self.motor_system._policy, SurfacePolicy):
            # For the surface-agent policy, update last action as if we have
            # just moved tangentially
            # Results in us seemlessly transitioning into the typical
            # corrective movements (forward or orientation) of the surface-agent
            # policy
            self.motor_system._policy.action = MoveTangentially(
                agent_id=self.motor_system._policy.agent_id,
                distance=0.0,
                direction=(0, 0, 0),
            )

            # TODO cleanup where this is performed, and make variable names more general
            # TODO also only log this when we are doing detailed logging
            # TODO M clean up these action details loggings; this may need to remain
            # local to a "motor-system buffer" given that these are model-free
            # actions that have nothing to do with the LMs
            # Store logging information about jump success
            self.motor_system._policy.action_details["pc_heading"].append("jump")
            self.motor_system._policy.action_details["avoidance_heading"].append(False)
            self.motor_system._policy.action_details["z_defined_pc"].append(None)

        else:
            self.get_good_view_with_patch_refinement()

    def handle_failed_jump(self, pre_jump_state, first_sensor):
        """Deal with the results of a failed hypothesis-testing jump.

        A failed jump is "off-object", i.e. the object is not perceived by the sensor.
        """
        logger.debug("No object visible from hypothesis jump, or inside object!")
        logger.debug("Returning to previous position")

        set_agent_pose = SetAgentPose(
            agent_id=self.motor_system._policy.agent_id,
            location=pre_jump_state["position"],
            rotation_quat=pre_jump_state["rotation"],
        )
        # All sensors are updated globally by actions, and are therefore
        # identical
        set_sensor_rotation = SetSensorRotation(
            agent_id=self.motor_system._policy.agent_id,
            rotation_quat=pre_jump_state["sensors"][first_sensor]["rotation"],
        )
        _, _ = self.step(set_agent_pose)
        self._observation, proprioceptive_state = self.step(set_sensor_rotation)

        assert np.all(
            proprioceptive_state[self.motor_system._policy.agent_id]["position"]
            == pre_jump_state["position"]
        ), "Failed to return agent to location"
        assert np.all(
            proprioceptive_state[self.motor_system._policy.agent_id]["rotation"]
            == pre_jump_state["rotation"]
        ), "Failed to return agent to orientation"

        for current_sensor in proprioceptive_state[self.motor_system._policy.agent_id][
            "sensors"
        ].keys():
            assert np.all(
                proprioceptive_state[self.motor_system._policy.agent_id]["sensors"][
                    current_sensor
                ]["rotation"]
                == pre_jump_state["sensors"][current_sensor]["rotation"]
            ), "Failed to return sensor to orientation"

        self.motor_system._state = MotorSystemState(proprioceptive_state)

        # TODO explore reverting to an attempt with touch_object here,
        # only moving back to our starting location if this is unsuccessful
        # after e.g. 16 glances around where we arrived; NB however that
        # if we're inside the object, then we don't want to do this


class OmniglotDataLoader(EnvironmentDataLoaderPerObject):
    """Dataloader for Omniglot dataset."""

    def __init__(
        self,
        alphabets,
        characters,
        versions,
        env: EmbodiedEnvironment,
        motor_system: MotorSystem,
        rng,
        transform=None,
        *args,
        **kwargs,
    ):
        """Initialize dataloader.

        Args:
            alphabets: List of alphabets.
            characters: List of characters.
            versions: List of versions.
            env: An instance of a class that implements :class:`EmbodiedEnvironment`.
            motor_system: The motor system.
            rng: Random number generator to use.
            transform: Callable used to transform the observations returned
                 by the environment.

            *args: Additional arguments
            **kwargs: Additional keyword arguments

        Raises:
            TypeError: If `motor_system` is not an instance of `MotorSystem`.
        """
        if not isinstance(motor_system, MotorSystem):
            raise TypeError(
                f"motor_system must be an instance of MotorSystem, got {motor_system}"
            )
        self.env = env
        self.rng = rng
        self.motor_system = motor_system
        self.transform = transform
        if self.transform is not None:
            for t in self.transform:
                if t.needs_rng:
                    t.rng = self.rng
        self._observation, proprioceptive_state = self.reset()
        self.motor_system._state = (
            MotorSystemState(proprioceptive_state) if proprioceptive_state else None
        )
        self._action = None
        self._counter = 0

        self.alphabets = alphabets
        self.characters = characters
        self.versions = versions
        self.current_object = 0
        self.n_objects = len(characters)
        self.episodes = 0
        self.epochs = 0
        self.primary_target = None
        self.object_names = [
            str(self.env.alphabet_names[alphabets[i]]) + "_" + str(self.characters[i])
            for i in range(self.n_objects)
        ]

    def post_episode(self):
        self.motor_system.post_episode()
        self.cycle_object()
        self.episodes += 1

    def post_epoch(self):
        self.epochs += 1

    def cycle_object(self):
        """Switch to the next character image."""
        next_object = (self.current_object + 1) % self.n_objects
        logger.info(
            f"\n\nGoing from {self.current_object} to {next_object} of {self.n_objects}"
        )
        self.change_object_by_idx(next_object)

    def change_object_by_idx(self, idx):
        """Update the object in the scene given the idx of it in the object params.

        Args:
            idx: Index of the new object and ints parameters in object params
        """
        assert idx <= self.n_objects, "idx must be <= self.n_objects"
        self.env.switch_to_object(
            self.alphabets[idx], self.characters[idx], self.versions[idx]
        )
        self.current_object = idx
        self.primary_target = {
            "object": self.object_names[idx],
            "rotation": np.quaternion(0, 0, 0, 1),
            "euler_rotation": np.array([0, 0, 0]),
            "quat_rotation": [0, 0, 0, 1],
            "position": np.array([0, 0, 0]),
            "scale": [1.0, 1.0, 1.0],
        }


class SaccadeOnImageDataLoader(EnvironmentDataLoaderPerObject):
    """Dataloader for moving over a 2D image with depth channel."""

    def __init__(
        self,
        scenes,
        versions,
        env: EmbodiedEnvironment,
        motor_system: MotorSystem,
        rng,
        transform=None,
        *args,
        **kwargs,
    ):
        """Initialize dataloader.

        Args:
            scenes: List of scenes
            versions: List of versions
            env: An instance of a class that implements :class:`EmbodiedEnvironment`.
            motor_system: The motor system.
            rng: Random number generator to use.
            transform: Callable used to transform the observations returned by
                the environment.
            *args: Additional arguments
            **kwargs: Additional keyword arguments

        Raises:
            TypeError: If `motor_system` is not an instance of `MotorSystem`.
        """
        if not isinstance(motor_system, MotorSystem):
            raise TypeError(
                f"motor_system must be an instance of MotorSystem, got {motor_system}"
            )
        self.env = env
        self.rng = rng
        self.motor_system = motor_system
        self.transform = transform
        if self.transform is not None:
            for t in self.transform:
                if t.needs_rng:
                    t.rng = self.rng
        self._observation, proprioceptive_state = self.reset()
        self.motor_system._state = (
            MotorSystemState(proprioceptive_state) if proprioceptive_state else None
        )
        self._action = None
        self._counter = 0

        self.scenes = scenes
        self.versions = versions
        self.object_names = self.env.scene_names
        self.current_scene_version = 0
        self.n_versions = len(versions)
        self.episodes = 0
        self.epochs = 0
        self.primary_target = None

    def post_episode(self):
        self.motor_system.post_episode()
        self.cycle_object()
        self.episodes += 1

    def post_epoch(self):
        self.epochs += 1

    def cycle_object(self):
        """Switch to the next scene image."""
        next_scene = (self.current_scene_version + 1) % self.n_versions
        logger.info(
            f"\n\nGoing from {self.current_scene_version} to {next_scene} of "
            f"{self.n_versions}"
        )
        self.change_object_by_idx(next_scene)

    def change_object_by_idx(self, idx):
        """Update the object in the scene given the idx of it in the object params.

        Args:
            idx: Index of the new object and ints parameters in object params
        """
        assert idx <= self.n_versions, "idx must be <= self.n_versions"
        logger.info(
            f"changing to obj {idx} -> scene {self.scenes[idx]}, version "
            f"{self.versions[idx]}"
        )
        self.env.switch_to_object(self.scenes[idx], self.versions[idx])
        self.current_scene_version = idx
        # TODO: Currently not differentiating between different poses/views
        target_object = self.object_names[self.scenes[idx]]
        # remove scene index from name
        target_object_formatted = "_".join(target_object.split("_")[1:])
        self.primary_target = {
            "object": target_object_formatted,
            "rotation": np.quaternion(0, 0, 0, 1),
            "euler_rotation": np.array([0, 0, 0]),
            "quat_rotation": [0, 0, 0, 1],
            "position": np.array([0, 0, 0]),
            "scale": [1.0, 1.0, 1.0],
        }


class SaccadeOnImageFromStreamDataLoader(SaccadeOnImageDataLoader):
    """Dataloader for moving over a 2D image with depth channel."""

    def __init__(
        self,
        env: EmbodiedEnvironment,
        motor_system: MotorSystem,
        rng,
        transform=None,
        *args,
        **kwargs,
    ):
        """Initialize dataloader.

        Args:
            env: An instance of a class that implements :class:`EmbodiedEnvironment`.
            motor_system: The motor system.
            rng: Random number generator to use.
            transform: Callable used to transform the observations returned by
                the environment.
            *args: Additional arguments
            **kwargs: Additional keyword arguments

        Raises:
            TypeError: If `motor_system` is not an instance of `MotorSystem`.
        """
        if not isinstance(motor_system, MotorSystem):
            raise TypeError(
                f"motor_system must be an instance of MotorSystem, got {motor_system}"
            )
        # TODO: call super init instead of duplication code & generally clean up more
        self.env = env
        self.rng = rng
        self.motor_system = motor_system
        self.transform = transform
        if self.transform is not None:
            for t in self.transform:
                if t.needs_rng:
                    t.rng = self.rng
        self._observation, proprioceptive_state = self.reset()
        self.motor_system._state = (
            MotorSystemState(proprioceptive_state) if proprioceptive_state else None
        )
        self._action = None
        self._counter = 0
        self.current_scene = 0
        self.episodes = 0
        self.epochs = 0
        self.primary_target = None

    def pre_epoch(self):
        # TODO: Could give a start index as parameter
        self.change_scene_by_idx(0)

    def post_episode(self):
        self.motor_system.post_episode()
        self.cycle_scene()
        self.episodes += 1

    def post_epoch(self):
        self.epochs += 1

    def cycle_scene(self):
        """Switch to the next scene image."""
        next_scene = self.current_scene + 1
        logger.info(f"\n\nGoing from {self.current_scene} to {next_scene}")
        # TODO: Do we need a separate method for this ?
        self.change_scene_by_idx(next_scene)

    def change_scene_by_idx(self, idx):
        """Update the object in the scene given the idx of it in the object params.

        Args:
            idx: Index of the new object and ints parameters in object params
        """
        logger.info(f"changing to scene {idx}")
        self.env.switch_to_scene(idx)
        self.current_scene = idx
        # TODO: Currently not differentiating between different poses/views
        # TODO: Are the targets important here ? How can we provide the proper
        # targets corresponding to the current scene ?
        self.primary_target = {
            "object": "no_label",
            "rotation": np.quaternion(0, 0, 0, 1),
            "euler_rotation": np.array([0, 0, 0]),
            "quat_rotation": [0, 0, 0, 1],
            "position": np.array([0, 0, 0]),
            "scale": [1.0, 1.0, 1.0],
        }
