# Copyright 2025 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import logging

import numpy as np

from tbp.monty.frameworks.models.abstract_monty_classes import GoalStateGenerator
from tbp.monty.frameworks.models.states import GoalState
from tbp.monty.frameworks.utils.communication_utils import get_state_from_channel

logger = logging.getLogger(__name__)


class GraphGoalStateGenerator(GoalStateGenerator):
    """Generate sub-goal states until the received goal state is achieved.

    A component associated with each learning module that receives a high level goal
    state, and generates sub-goal states until the received goal state is achieved.

    Generated goal-states are received by either:
        i) other learning modules, which may model world-objects (e.g. a mug), or may
        model internal systems (e.g. the agent's robotic limb)
        ii) motor actuators, in which case they represent simpler, primitive goal-states
        for the actuator to achieve (e.g. location and orientation of an actuator-sensor
        pair)

    As well as the high-level, "driving" goal-state, generated goal-states can also be
    conditioned on other information, such as the LMs current most-likely hypothesis,
    and the structure of known object models (i.e. information local to the LM).

    Note all goal-states conform to the State-class cortical messaging protocol (CMP).
    """

    def __init__(self, parent_lm, goal_tolerances=None, **kwargs) -> None:
        """Initialize the GSG.

        Args:
            parent_lm: The learning-module class instance that the GSG is embedded
                within.
            goal_tolerances: The tolerances for each attribute of the goal-state
                that can be used by the GSG when determining whether a goal-state is
                achieved. These are not necessarily the same as an LM's tolerances used
                for matching, as here we are evaluating whether a goal-state is
                achieved.
            **kwargs: Additional keyword arguments. Unused.
        """
        self.parent_lm = parent_lm
        if goal_tolerances is None:
            self.goal_tolerances = dict(
                location=0.015,  # distance in meters
            )
        else:
            self.goal_tolerances = goal_tolerances

        self.reset()
        self.set_driving_goal_state(self._generate_none_goal_state())

    # =============== Public Interface Functions ===============

    # ------------------ Getters & Setters ---------------------

    def reset(self):
        """Reset any stored attributes of the GSG."""
        self.set_driving_goal_state(self._generate_none_goal_state())
        self._set_output_goal_state(self._generate_none_goal_state())
        self.parent_lm.buffer.update_stats(
            dict(
                goal_states=[],
                matching_step_when_output_goal_set=[],
                goal_state_achieved=[],
            ),
            update_time=False,
            append=False,
            init_list=False,
        )

    def set_driving_goal_state(self, received_goal_state):
        """Receive a new high-level goal to drive this goal-state-generator (GSG).

        If none is provided, the goal-state generator should default to pursuing a
        goal-state of high confidence, with no other attributes of the state specified;
        in other words, it attempts to reduce uncertainty about the LM's output
        (object ID and pose, whatever these may be).

        TODO M: Currently GSGs always use the default, however future
        work will implement hierarchical action policies/GSGs, as well as the ability to
        specify a top goal-state by the experimenter.

        TODO M : we currently just use "None" as a placehodler for the default
        goal-state
        > plan : set the default driving goal-state to a meaningful, non-None value
        that is compatible with the current method for checking convergence of an LM,
        such that achieving the driving goal-state can be used as a test for Monty
        convergence. This might be something like the below.
        """
        # if received_goal_state is None:
        #     # The current default goal-state, which is to reduce uncertainty; this is
        #     # defined by having a high-confidence in the goal-state, and an arbitrary
        #     # single object ID.
        #     self.driving_goal_state = GoalState(
        #         location=None,
        #         morphological_features=None,
        #         non_morphological_features={
        #             "object_id": "*",  # Match any object so long as it is described
        #             # by a single ID
        #             "location_rel_model": None
        #         },
        #         confidence=1.0,  # Should have high confidence
        #         use_state=False,
        #         sender_id=self.parent_lm.learning_module_id,
        #         sender_type="GSG",
        #         goal_tolerances=None,
        #     )
        # else:

        self.driving_goal_state = received_goal_state

    def output_goal_states(self) -> list[GoalState]:
        """Retrieve the output goal-states of the GSG.

        This is the goal-state projected to other LM's GSGs +/- motor-actuators.

        Returns:
            Output goal-states of the GSG if it exists, otherwise empty list.
        """
        return [self.output_goal_state] if self.output_goal_state else []

    # ------------------- Main Algorithm -----------------------

    def step(self, observations):
        """Step the GSG.

        Check whether the GSG's output and driving goal-states are achieved, and
        generate a new output goal-state if necessary.
        """
        output_goal_achieved = self._check_output_goal_state_achieved(observations)

        self._update_gsg_logging(output_goal_achieved)

        # If driving goal-state achieved, from this LM's perspective, other LMs/GSGs
        # need not do anything
        # TODO M re-introduce when replacing the current check for convergence with
        # achievement of the driving goal-state
        # if self._check_driving_goal_state_achieved():
        #     self._set_output_goal_state(
        #         new_goal_state=self._generate_none_goal_state()
        #     )
        # else:
        #     # Below code block

        if self._check_need_new_output_goal(output_goal_achieved):
            self._set_output_goal_state(
                new_goal_state=self._generate_goal_state(observations)
            )
        elif self._check_keep_current_output_goal():
            pass
        else:
            self._set_output_goal_state(new_goal_state=self._generate_none_goal_state())

    # ======================= Private ==========================

    # ------------------- Main Algorithm -----------------------

    def _generate_none_goal_state(self):
        """Return a None-type goal state.

        A None-type goal state specifies nothing other than high confidence.

        NOTE currently we just use a None value, however in the future we might specify
        a GoalState object with a None value for the location, morphological features,
        etc, or some variation of this.
        """
        return None

    def _generate_goal_state(self, observations):
        """Generate a new goal-state to send out to other LMs and/or motor actuators.

        Given the driving goal-state, and information from the parent LM of the GSG
        (including the current observations), generate a new goal-state to send out
        to other LMs and/or motor actuators.

        Note the output goal-state is in a common, body-centered frame of reference,
        as for voting, such that different modules can mutually communicate.

        This version is a base place-holder method that just returns a None goal-state,
        and does not actually make use of observations or the driving-goal state.

        Returns:
            A None-type goal state.
        """
        return self._generate_none_goal_state()

    def _check_states_different(
        self,
        state_a,
        state_b,
        diff_tolerances,
    ) -> bool:
        """Check whether two states are different.

        States need to be different only by one feature/dimension to be considered
        different.

        When checking whether the states are different, a dictionary must be passed
        of tolerances; in the GSG-class, this is typically the GSG's own default
        goal-tolerances that are used, but specific tolerances can also be passed
        along with a goal-state itself (i.e. achieve this goal-state, to within these
        tolerance bounds).

        Note:
            If a state is undefined (None), we define a difference as unmeaningful and
            therefore return False. Similarly, for any feature of a state (or dimension
            of a feature) that is undefined (None or NaN), we do not return any
            difference along that dimension.

        TODO M consider making this a utility function, as might be useful in e.g. the
        LM itself as well. However, the significant presence of None/NaN values in
        Goal States may mean we want to take a different appraoch.

        Returns:
            Whether the states are different.
        """
        if state_a is None or state_b is None:
            return False

        states_different = False
        for tolerance_key, tolerance_val in diff_tolerances.items():
            # TODO M implement feature comparisons for other State features (e.g.
            # confidence)
            # TODO M consider using the LM's lm.tolerances for the default values of
            # diff_tolerances

            if tolerance_key == "location":
                if state_a.location is not None and state_b.location is not None:
                    distance = np.linalg.norm(state_a.location - state_b.location)

            elif tolerance_key == "pose_vectors":
                if (
                    state_a.morphological_features is not None
                    and state_b.morphological_features is not None
                ):
                    raise NotImplementedError(
                        "TODO M implement pose-vector comparisons that handle "
                        "symmetry of objects"
                    )
                    # TODO M consider using an angular distance instead of Euclidean
                    # when we actually begin making use of this feature; try to ensure
                    # this handles symmetry conditions e.g. flipped principal curvature
                    # directions.
                    distance = self._euc_dist_ignoring_nan(
                        state_a.morphological_features["pose_vectors"],
                        state_b.morphological_features["pose_vectors"],
                    )

            states_different = distance > tolerance_val
            if states_different:
                return states_different

        return states_different

    def _check_driving_goal_state_achieved(self) -> bool:
        """Check if parent LM's output state is close enough to driving goal-state.

        TODO M Move some of the checks for convergence here

        Returns:
            Whether the parent LM's output state is close enough to the driving
            goal-state.
        """
        if self.driving_goal_state.goal_tolerances is None:
            # When not specified by the incoming driving-goal-state, use the GSG's own
            # default matching tolerances
            diff_tolerances = self.goal_tolerances

        return self._check_states_different(
            self.parent_lm.get_output(), self.driving_goal_state, diff_tolerances
        )

    def _check_output_goal_state_achieved(self, observations) -> bool:
        """Check if the output goal-state was achieved.

        Check whether the information entering the LM suggests that the output goal
        state of the GSG was achieved. Recall that the output goal is the one sent by
        this GSG to other LMs and motor-actuators to be achieved.

        Note:
            In the future we might use feedback from a receiving system that is not
            "sensory input" (i.e. does not inform the graph building of this parent LM);
            Instead, such feedback could include the state of an LM that controls a
            motor system (such as a hand model LM), or the state of a motor-actuator
            (akin to proprioceptive feedback); in this case, we could directly compare
            the output goal-state to the state recived by this feedback, to determine
            whether the goal-state was likely achieved. This input would likely come
            from a separate channel (similar to voting). Finally, note that this
            information could be complimentary to feedback from the sensory input and
            our sensory predictions, as in some cases we might have no proprioceptive
            feedback, while in other cases we might have no sensory input (e.g.
            blindfolded); alignment or mismatch between these two could form useful
            signals for learning policies and object behaviors.

        Returns:
            Whether the output goal-state was achieved.
        """
        if self.output_goal_state is not None:
            goal_achieved = self._check_input_matches_sensory_prediction(observations)

            return goal_achieved

        else:
            return False

    def _check_input_matches_sensory_prediction(self, observations):
        """Check whether the input matches the sensory prediction.

        Here the sensory prediction is simply that the input state has changed, as
        when the motor-system attempts to achieve a goal-state and fails (e.g. due
        to collision with another object), it moves back to the original position.

        Note that there can still be some difference even when a goal-state
        failed, as the feature-change-SM and motor-only steps can result in the agent
        moving after it has returned to its original position. Futhermore, there may
        not always be a difference if the agent did "succeed", if the goal-state
        it wanted to acheive happened to be very close to its original position. Thus
        this is an approximate method.

        TODO M implement also using the target goal-state and internal model
        to predict a specific input state, and then compare to that to determine
        not just whether a movement took place, but whether the agent moved to a
        particular point on a particular object.

        Returns:
            Whether the input matches the sensory prediction.
        """
        sensor_channel_name = self.parent_lm.buffer.get_first_sensory_input_channel()

        current_sensory_input = get_state_from_channel(
            states=observations, channel_name=sensor_channel_name
        )

        prev_input_states = self.parent_lm.buffer.get_previous_input_states()
        if prev_input_states is not None:
            previous_sensory_input = get_state_from_channel(
                states=prev_input_states,
                channel_name=sensor_channel_name,
            )
        else:
            previous_sensory_input = None
        # NB if no history of inputs, get_previous_input_states returns None, in which
        # case _check_states_different will return False, and we return goal_achieved as
        # False, as we cannot meaningfully evaluate whether this occured

        input_changed = self._check_states_different(
            current_sensory_input,
            previous_sensory_input,
            diff_tolerances=self.goal_tolerances,
        )

        return input_changed

    def _check_need_new_output_goal(self, output_goal_achieved) -> bool:
        """Determine whether the GSG should generate a new output goal-state.

        In the base version, this is True if the output-goal was achieved, suggesting
        we should move on to the next goal.

        Returns:
            Whether the GSG should generate a new output goal-state.
        """
        if output_goal_achieved:
            return True
        else:
            return False

    def _check_keep_current_output_goal(self) -> bool:
        """Should we keep our current goal?

        If we don't need a new goal, determine whether we should keep our current
        goal (as opposed to output no goal at all).

        Returns:
            Whether we should keep our current goal.
        """
        return True

    def _euc_dist_ignoring_nan(self, a, b):
        """Euclidean distance between two arrays, ignoring NaN values.

        Take the Euclidean distance between two arrays, but only measuring the
        distance where both arrays have non-NaN values.

        Args:
            a: First array
            b: Second array

        Returns:
            Euclidean distance between the two arrays, ignoring NaN values; if all
            values are NaN, return 0

        TODO M consider making a general utility function
        """
        assert a.shape == b.shape, "Arrays must be of the same shape"

        mask = ~np.isnan(a) & ~np.isnan(b)

        # If the mask is empty, return 0 (i.e. there is no meaningful distance
        # between the two vectors)
        if len(mask) == 0:
            return 0

        else:
            return np.linalg.norm(a[mask] - b[mask])

    # ------------------ Getters, Setters & Logging ---------------------

    def _set_output_goal_state(self, new_goal_state):
        """Set the output goal-state of the GSG."""
        self.output_goal_state = new_goal_state

    def _update_gsg_logging(self, output_goal_achieved: bool):
        """Update any logging information (stored in the parent LM's buffer).

        Update any logging information (stored in the parent LM's buffer), such as
        the matching step on which an output goal-state was output.
        """
        # Only consider output-state achieved for the purpose of logging when the
        # output-goal state is meaningful (i.e. not None)
        if self.output_goal_state is not None:
            # Subtract 1 as the goal-state was actually set (and potentially achieved)
            # on the previous step, we are simply first checking it now

            match_step = self.parent_lm.buffer.get_num_matching_steps() - 1
            self.output_goal_state.info["achieved"] = output_goal_achieved
            self.output_goal_state.info["matching_step_when_output_goal_set"] = (
                match_step
            )
            self.parent_lm.buffer.update_stats(
                dict(
                    goal_states=self.output_goal_state,
                    matching_step_when_output_goal_set=match_step,
                    goal_state_achieved=output_goal_achieved,
                ),
                update_time=False,
                append=True,
                init_list=True,
            )


class EvidenceGoalStateGenerator(GraphGoalStateGenerator):
    """Generator of goal states for an evidence-based graph LM.

    GSG specifically setup for generating goal states for an evidence-based graph LM,
    which can therefore leverage the hypothesis-testing action policy. This policy uses
    hypotheses about the most likely objects, as well as knowledge of their structure
    from long-term memory, to propose test-points that should efficiently disambiguate
    the ID or pose of the object the agent is currently observing.

    TODO M separate out the hypothesis-testing policy (which is one example of a
    model-based policy), from the GSG, which is the system that is capable of leveraging
    a variety of model-based policies.
    """

    def __init__(
        self,
        parent_lm,
        goal_tolerances=None,
        elapsed_steps_factor=10,
        min_post_goal_success_steps=np.inf,
        x_percent_scale_factor=0.75,
        desired_object_distance=0.03,
        wait_growth_multiplier=2,
        **kwargs,
    ) -> None:
        """Initialize the Evidence GSG.

        Args:
            parent_lm: ?
            goal_tolerances: ?
            elapsed_steps_factor: Factor that considers the number of elapsed
                steps as a possible condition for initiating a hypothesis-testing goal
                state; should be set to an integer reflecting a number of steps. In
                general, when we have taken number of non-goal-state driven steps
                greater than elapsed_steps_factor, then this is an indiciation to
                initiate a hypothesis-testing goal-state. In addition however, we can
                multiply elapsed_steps_factor by an exponentiall increasing wait-factor,
                such that we use longer and longer intervals as the experiment
                continues. Defaults to 10.
            min_post_goal_success_steps: Number of necessary steps for a hypothesis
                goal-state to be considered. Unlike elapsed_steps_factor, this is a
                *necessary* criteria for us to generate a new hypothesis-testing
                goal-state. For example, if set to 5, then the agent must take 5
                non-hypothesis-testing steps before it can even consider generating a
                new hypothesis-testing goal-state. Infinity by default, resulting in no
                use of the hypothesis-testing policy (desirable for unit tests etc.).
                Defaults to np.infty.
            x_percent_scale_factor: Scale x-percent threshold to decide when to focus
                on pose rather than determining object ID; in particular, this is used
                to determine whether the top object is sufficiently more likely (based
                on MLH evidence) than the second MLH object to warrant focusing on
                disambiguating the pose of the first; should be bounded between 0:1.0.
                If x_percent_scale_factor=1.0, then will wait until the standard
                x-percent threshold is exceeded, equivalent to the LM converging to a
                single object, but not a pose. If it is <1.0, then we will start testing
                pose of the MLH object even before we are entirely certain about its ID.
                Defaults to 0.75.
            desired_object_distance: The desired distance between the agent and the
                object, which is used to determine whether the agent is close enough to
                the object to consider it "achieved". Note this need not be the same as
                the one specified for the motor-system (e.g. the surface-policy), as we
                may want to aim for an initially farther distance, while the
                surface-policy may want to stay quite close to the object. Defaults to
                0.03.
            wait_growth_multiplier: Multiplier used to increase the `wait_factor`, which
                in turn controls how long to wait before the next jump attempt.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(parent_lm, goal_tolerances, **kwargs)

        self.elapsed_steps_factor = elapsed_steps_factor
        self.min_post_goal_success_steps = min_post_goal_success_steps
        self.x_percent_scale_factor = x_percent_scale_factor
        self.desired_object_distance = desired_object_distance
        self.wait_growth_multiplier = wait_growth_multiplier

    # ======================= Public ==========================

    # ------------------- Main Algorithm -----------------------

    def reset(self):
        """Reset additional parameters specific to the Evidence GSG."""
        super().reset()

        self.focus_on_pose = False  # Whether the jump should be executed to focus on
        # distinguishing between possible poses of the current MLH object, rather
        # than trying to distinguish different possible object IDs.
        self.wait_factor = 1  # Initial value; scales how long to wait before the
        # next jump attempt
        self.prev_top_mlhs = None  # Store the top two object hypothesis IDs from
        # previous hypothesis-testing actions; used to track when these have changed,
        # and therefore a possible reason to initiate another hypothesis-testing action.
        # TODO M consider moving to buffer.

    # ======================= Private ==========================

    # ------------------- Main Algorithm -----------------------

    def _generate_goal_state(self, observations) -> list:
        """Use the hypothesis-testing policy to generate a goal-state.

        The goal-state will rapidly disambiguate the pose and/or ID of the object the
        LM is currently observing.

        Returns:
            A goal-state for the motor system.
        """
        # Determine where we want to test in the MLH graph
        target_loc_id, target_separation = self._compute_graph_mismatch()

        # Get pose information for the target point
        target_info = self._get_target_loc_info(target_loc_id)

        # Estimate how important this goal-state will be for the Monty-system as a
        # whole
        goal_confidence = self._compute_goal_confidence(
            lm_output_confidence=self.parent_lm.get_output().confidence,
            separation=target_separation,
        )

        # Compute the goal-state (for the motor-actuator)
        motor_goal_state = self._compute_goal_state_for_target_loc(
            observations,
            target_info,
            goal_confidence=goal_confidence,
        )

        return motor_goal_state

    def _compute_graph_mismatch(self):
        """Propose a point for the model to test.

        The aim is to propose a point for the model to test, with the aim of performing
        object pose and ID recognition, by looking at the graph of the most likely
        object, and comparing it to the graph of the second most likely object. If there
        is a local mismatch in the graphs (e.g. the presense of a handle in one and not
        the other), this should return the necessary coordinates to move there.

        Returns the index of the point in the graph of the most likely object that
        should be tested, along with the size of the L-2 separation of this point to
        the nearest point in the graph of the second most likely object.

        --- Some Details ---
        Part of this method transforms the graph of the most likely object into
        the reference frame in which the 2nd most-likely object was *learned*.
        We perform this operation because when comparing the two point-clouds (using
        the points of the most-likely object as queries), we can then re-use the KDTree
        already constructed for the 2nd most-likely object, hence the importance of
        being in that reference frame

        TODO M eventually can try looking at more objects, or flipping the MLH
        object - e.g. if the two most likely are the mug and one of the handle-less
        cups, then depending on the order in which we compare them, we may not
        actually identify the handle as a good candidate to test (i.e. if one graph is
        entirely a subset of the other graph). We could use the returned separation
        distance to estimate which of these approaches would be better.
        - i.e. imagine 2nd MLH is a mug with a handle, and MLH has no handle; when
        checking all the points for the handleless mug, there will always be nearby
        points
        - Re. implementing this : could start with the MLH as the query points, looking
        for points with minimal neighbors with the 2nd most likely graph; if found
        too many neighbors in a given radius (threshold dependent), this suggest the 1st
        MLH graph is a sub-graph of the 2nd MLH; therefore, check whether the 2nd graph
        has any points with few neighbors with the first; if still many neighbors, this
        could then serve as a learning signal to merge the graphs? (at least at some
        levels of hierarchy) --> NB merging should use "picky" graph-building method to
        ensure we don't just double the number of points in the graph unecessarily

        TODO M consider adding a factor so that we ensure our testing spot is also far
        away from any previously visited locations (at least according to MLH path),
        including our current location.

        Returns:
            The index of the point in the model to test.
        """
        logger.debug("Proposing an evaluation location based on graph mismatch")

        top_id, second_id = self.parent_lm.get_top_two_mlh_ids()

        top_mlh = self.parent_lm.get_mlh_for_object(top_id)
        # Determine the second most-likely object for saving to history, even if we
        # are going to focus on pose mismatch
        second_mlh_object = self.parent_lm.get_mlh_for_object(second_id)

        top_mlh_graph = self.parent_lm.get_graph(top_id, input_channel="first").pos

        if self.focus_on_pose:
            # Overwrite the second most likely hypothesis with the second most likely
            # *pose* of the most-likely object
            second_id = top_id
            _, second_mlh = self.parent_lm.get_top_two_pose_hypotheses_for_graph_id(
                top_id
            )

        else:
            second_mlh = second_mlh_object

        # == Corrective transformation ==
        # Fully correct the origin and rotation of the top object's graph so it is in
        # the same reference frame as the second object's graph was learned
        # Note the graph of the second most likely object is already in the same
        # reference frame (i.e. the one from learning) used when constructing the KDTree

        # Convert to environmental coordinates, and normalize by current MLH location
        # TODO M refactor this into a function that can be reapplied to arbitrary graphs
        # Note the MLH rotation is the rotation required to match a displacement to
        # a model, so it is the *inverse* of e.g. the ground-truth rotation
        # TODO M: See if apply_rf_transform_to_points could be used here
        rotated_graph = top_mlh["rotation"].inv().apply(top_mlh_graph)
        current_mlh_location = top_mlh["rotation"].inv().apply(top_mlh["location"])
        top_mlh_graph = rotated_graph - current_mlh_location
        # Convert from environmental coordinates to the learned coordinate of 2nd object
        # Thus we don't need to invert the stored rotation, as we would like to actually
        # apply the inverse form.
        top_mlh_graph = (
            second_mlh["rotation"].apply(top_mlh_graph) + second_mlh["location"]
        )

        # Perform the same transformation to the estimated location (sanity check)
        transformed_current_loc = top_mlh["rotation"].inv().apply(
            top_mlh["location"]
        ) - top_mlh["rotation"].inv().apply(top_mlh["location"])
        transformed_current_loc = (
            second_mlh["rotation"].apply(transformed_current_loc)
            + second_mlh["location"]
        )
        assert np.all(transformed_current_loc == second_mlh["location"]), (
            "Graph transformation to 2nd object reference frame not returning correct "
            "transformed location"
        )

        # Perform kdtree search to identify the point with the most distant
        # nearest-neighbor
        # Note we ultimately want the target location to be one on the most likely
        # graph, so we pass the top-MLH graph in as the qeury points
        radius_node_dists = self.parent_lm.get_graph(
            second_id, input_channel="first"
        ).find_nearest_neighbors(
            top_mlh_graph,
            num_neighbors=1,
            return_distance=True,
        )

        target_loc_id = np.argmax(radius_node_dists)
        target_loc_separation = np.max(radius_node_dists)

        self.prev_top_mlhs = [top_mlh, second_mlh_object]

        return target_loc_id, target_loc_separation

    def _get_target_loc_info(self, target_loc_id):
        """Given a target location ID, get the target location and pose vectors.

        Note:
            Currently assumes we are computing with the MLH graph.

        Returns:
            A dictionary containing the hypothesis to test, the target location and
            surface normal of the target point on the object.
        """
        mlh = self.parent_lm.get_current_mlh()
        mlh_id = mlh["graph_id"]

        target_object = self.parent_lm.get_graph(mlh_id)
        sensor_channel_name = self.parent_lm.buffer.get_first_sensory_input_channel()
        target_graph = target_object[sensor_channel_name]
        target_loc = target_graph.pos[target_loc_id]
        surface_normal_mapping = target_graph.feature_mapping["pose_vectors"]
        target_surface_normal = target_graph.x[
            target_loc_id, surface_normal_mapping[0] : surface_normal_mapping[0] + 3
        ]

        target_info = {
            "hypothesis_to_test": mlh,
            "target_loc": target_loc,
            "target_surface_normal": target_surface_normal,
        }

        return target_info

    def _compute_goal_confidence(
        self, lm_output_confidence, separation, space_size=1.0, confidence_weighting=0.1
    ):
        """Calculate the confidence of the goal-state.

        The confidence is based on the e.g. separation in hypothesis-space between the
        two MLH, and the confidence associated with the MLH classificaiton of the parent
        LM. Currently just retuns the confidence of the parent LM but TODO M implement a
        more sophisticated function.

        TODO M How to normalize the displacement?
        Could put through a sigmoid, that is perhaps scaled by the size of the object?
        Could divide by e.g. the size of the object to make it likely to be <1, and
        then clip it; that way any subtle differences between LMs is likely to be
        preserved, i.e. rather than them all clipping to 1.0; can then just make
        sure this value is weighted heavily compared to confidence when computing
        the overall strenght of the goal-state.
        - size of the object could be estimated from the minimum and maximum corners
        - or use the max size of the graph --> Note this doesn't account for the
        actual size of the object, and these grid-models are not currently used

        Returns:
            The confidence of the goal-state.
        """
        # Provisional implementation:
        # squashed_displacement = np.clip(separation / space_size, 0, 1)
        # goal_confidence = squashed_displacement + confidence_weighting
        # * lm_output_confidence

        goal_confidence = lm_output_confidence

        return goal_confidence

    def _compute_goal_state_for_target_loc(
        self, observations, target_info, goal_confidence=1.0
    ) -> GoalState:
        """Specify a goal state for the motor-actuator.

        Based on a target location (in object-centric coordinates) and the associated
        surface normal of that location, specify a goal state for the motor-actuator,
        such that any sensors associated with the motor-actuator should be pointed down
        at and observing the target location (i.e. parallel to the surface normal).

        For the movement to have a high probability of arriving at the desired location,
        the current hypothesis of the object ID and pose used to inform the movement
        should be correct, although subsequent observations may still provide useful
        information to the agent, i.e. even if we are wrong about the object ID and
        pose.

        Args:
            observations: The current observations, which should include the sensory
                input.
            target_info: A dictionary containing the target location and surface normal
                of the target point on the object.
            goal_confidence: The confidence of the goal-state, which should be in the
                range [0, 1]. This is used by receiving modules to weigh the
                importance of the goal-state relative to other goal-states.

        Returns:
            A goal-state for the motor-actuator.
        """
        # Determine the displacement, and therefore the environmental target location,
        # that we will use
        sensor_channel_name = self.parent_lm.buffer.get_first_sensory_input_channel()
        sensory_input = get_state_from_channel(
            states=observations, channel_name=sensor_channel_name
        )
        displacement = (
            target_info["target_loc"] - target_info["hypothesis_to_test"]["location"]
        )

        object_rot = target_info["hypothesis_to_test"]["rotation"].inv()  # MLH rotation
        # is stored as the rotation needed to convert a displacement to the object pose,
        # so the *object pose* is given by its inverse

        # Rotate the displacement; note we're converting from an *internal* object frame
        # of reference, to the global frame of reference; thus, we rotate not by the
        # inverse, but by the actual object orientation.
        rotated_disp = object_rot.apply(displacement)

        # The target location on the object's surface in global/body-centric coordinates
        proposed_surface_loc = sensory_input.location + rotated_disp

        # Rotate the learned surface normal (which was commited to memory assuming a
        # default 0,0,0 orientation of the object)
        target_surface_normal_rotated = object_rot.apply(
            target_info["target_surface_normal"]
        )

        # Scale the surface normal by the desired distance x1.5 (i.e. so that we start
        # a bit further away from the object; we will separately move forward if we
        # are indeed facing it)
        surface_displacement = (
            target_surface_normal_rotated * self.desired_object_distance * 1.5
        )

        target_loc = proposed_surface_loc + surface_displacement

        # Extra metadata for logging. 'achieved' and
        # 'matching_step_when_output_goal_set' should be updated at the next step.
        # We initialize them as `None` to inidicate that no valid values have been set.
        info = {
            "proposed_surface_loc": proposed_surface_loc,
            "hypothesis_to_test": target_info["hypothesis_to_test"],
            "achieved": None,
            "matching_step_when_output_goal_set": None,
        }

        motor_goal_state = GoalState(
            location=np.array(target_loc),
            morphological_features={
                # Note the hypothesis-testing policy does not specify the roll of the
                # agent, because this is not relevant to the task
                "pose_vectors": np.array(
                    [
                        (-1) * target_surface_normal_rotated,
                        [np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan],
                    ]
                ),
                "pose_fully_defined": None,
                "on_object": 1,
            },
            non_morphological_features=None,
            confidence=goal_confidence,
            use_state=True,
            sender_id=self.parent_lm.learning_module_id,
            sender_type="GSG",
            goal_tolerances=None,
            info=info,
        )

        # TODO M consider also using the below sensor-predicted state as an additional
        # evalaution of how much we have achieved our goal, i.e. consistent with the
        # object we thought we were on; could have more detailed information using the
        # internal object model
        # sensor_predicted_state = GoalState(
        #     location=np.array(proposed_surface_loc),
        #     morphological_features={
        #         # Note the hypothesis-testing policy does not specify the roll of the
        #         # agent, because this is not relevant to the task
        #         "pose_vectors": np.array(
        #             [
        #                 target_surface_normal_rotated,
        #                 [np.nan, np.nan, np.nan],
        #                 [np.nan, np.nan, np.nan],
        #             ]
        #         ),
        #         "pose_fully_defined": None,
        #         "on_object": 1,
        #     },
        #     non_morphological_features=None,
        #     confidence=goal_confidence,
        #     use_state=True,
        #     sender_id=self.parent_lm.learning_module_id,
        #     sender_type="GSG",
        #     goal_tolerances=None,
        # )

        return motor_goal_state

    def _check_need_new_output_goal(self, output_goal_achieved) -> bool:
        """Determine whether the GSG should generate a new output goal-state.

        Unlike the base version, success in achieving the goal-state is not an
        indication to need a new goal, because we should now be exploring a new
        part of the hypothesis-space, and so want to stay there for some time.

        Returns:
            Whether the GSG should generate a new output goal-state.
        """
        if output_goal_achieved:
            return False

        else:
            return self._check_conditions_for_hypothesis_test()

    def _check_keep_current_output_goal(self) -> bool:
        """Determine whether the GSG should keep the current goal-state.

        Hypothesis-testing actions should be executed as one-off attempts, lest we get
        stuck in a loop of trying to achieve the same goal that is impossible (e.g.
        due to collision with objects).

        Returns:
            Whether the GSG should keep the current goal-state. Always returns False.
        """
        return False

    def _check_conditions_for_hypothesis_test(self):
        """Check if good chance to discriminate between conflicting object IDs or poses.

        Evaluates possible conditions for performing a hypothesis-guided action for
        pose and object ID determination, i.e. determines whether there is a good chance
        of discriminating between conflicting object IDs or poses.

        The schedule is designed to balance discriminating the pose and objects as
        efficiently as possible; TODO M future work can use the schedule conditions as
        primitives and use RL or evolutionary algorithms to optimize the relevant
        parameters.

        TODO M each of the below conditions could be their own method; could then pass
        a set of keys which we iterate through, and thereby quickly test as a
        hyper-parameter which of these are worth keeping, and which of these
        we should get rid of.

        Returns:
            Whether there's a good chance to discriminate between conflicting object IDs
            or poses.
        """
        num_elapsed_steps = self._get_num_steps_post_output_goal_generated()

        self.focus_on_pose = False  # Default

        if num_elapsed_steps <= self.min_post_goal_success_steps:
            # Exceeding this threshold is necessary to consider a jump
            return False

        # === Collect additional information that will be used to check conditions
        # for initializing a jump ===

        top_id, second_id = self.parent_lm.get_top_two_mlh_ids()

        # This happens when all hypothesis spaces are empty
        if top_id is None and second_id is None:
            return False

        if second_id is None:
            # If we only have one object with a single hypothesis, we should not
            # attempt to generate a goal state.
            if len(self.parent_lm.hyp_evidences_for_object(top_id)) == 1:
                return False

            # If we only know (i.e. have learned) about one object, we can focus on pose
            # In this case, get_top_two_mlh_ids returns the same IDs for top_id and
            # second_id
            self.focus_on_pose = True
            return True

        # Used to check if pose for top MLH has changed
        top_mlh = self.parent_lm.get_current_mlh()

        # If the MLH evidence is significantly above the second MLH (where "significant"
        # is determined by x_percent_scale_factor below), then focus on discriminating
        # its pose on some (random) occasions; always focus on pose if we've convereged
        # to one object
        # TODO M update so that not accessing private methods here; part of 2nd phase
        # of refactoring
        pm_base_thresh = self.parent_lm._threshold_possible_matches()
        pm_smaller_thresh = self.parent_lm._threshold_possible_matches(
            x_percent_scale_factor=self.x_percent_scale_factor
        )

        if (
            len(pm_smaller_thresh) == 1 and (self.parent_lm.rng.uniform() <= 0.5)
        ) or len(pm_base_thresh) == 1:
            # If we only have one object with a single hypothesis, we should not
            # attempt to generate a goal state.
            if len(self.parent_lm.hyp_evidences_for_object(top_id)) == 1:
                return False

            # We always focus on pose if there is just 1 possible match - if we are part
            # of the way towards being certain about the ID
            # (len(pm_smaller_thresh) == 1), then we sometimes (hence the randomness)
            # focus on pose.
            logger.debug(
                "Hypothesis jump indicated: One object more likely, focusing on pose"
            )
            self.focus_on_pose = True
            return True

        # If the identities or *order* (i.e. which one is most likely)
        # of the top two MLH changes, perform a new jump test, as this
        # is a reasonable heuristic for us having a new interesting
        # place to test
        # TODO when optimizing, consider using np.any rather than np.all, i.e. as long
        # as there is any change in the top two MLH
        elif self.prev_top_mlhs is not None and np.all(
            [
                self.prev_top_mlhs[0]["graph_id"],
                self.prev_top_mlhs[1]["graph_id"],
            ]
            != [top_id, second_id]
        ):
            logger.debug(
                "Hypothesis jump indicated: change or shuffle in top-two MLH IDs"
            )
            return True

        # If the most-likely pose of the top object has changed (e.g. we didn't find
        # the mug handle following a previous jump, and have therefore eliminated a
        # pose), then we are likely to gain new information by performing another jump
        # TODO expand this to handle change in translationm/location pose as well
        # TODO add a parameter that specifies the angle between the two poses above
        # which we consider it a new pose (rather than it needing to be identical)
        elif self.prev_top_mlhs is not None and np.all(
            top_mlh["rotation"].as_euler("xyz")
            != self.prev_top_mlhs[0]["rotation"].as_euler("xyz")
        ):
            logger.debug(
                "Hypothesis jump indicated: change in most-likely rotation of MLH"
            )
            return True

        # Otherwise, if a sufficient number of steps have elapsed,
        # still perform a jump; note however that this threshold exponentially
        # increases, so that we avoid continuously returning to the same location
        elif num_elapsed_steps % (self.wait_factor * self.elapsed_steps_factor) == 0:
            logger.debug(
                "Hypothesis jump indicated: sufficient steps elapsed with no jump"
            )

            self.wait_factor *= self.wait_growth_multiplier
            return True

        else:
            return False

    def _get_num_steps_post_output_goal_generated(self):
        """Number of steps since last output goal-state.

        Returns:
            The number of Monty-matching steps that have elapsed since the last time
            an output goal-state was generated.
        """
        return self.parent_lm.buffer.get_num_steps_post_output_goal_generated()
