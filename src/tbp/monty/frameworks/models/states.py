# Copyright 2025 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import numpy as np


class State:
    """State class used as message packages passed in Monty using CMP.

    The cortical messaging protocol (CMP) is used to pass messages between Monty
    components and makes sure we can easily set up arbitrary configurations of them.
    This class makes it easier to define the CMP in one place and defines the content
    and structure of messages passed between Monty components. It also contains some
    helper funtions to access and modify the message content.

    States are represented in this format but can be interpreted by the receiver in
    different ways:
       Observed states: states output py sensor modules
       Hypothesized states: states output by learning modules
       Goal states: motor output of learning modules

    Attributes:
        location: 3D vector representing the location of the state
        morphological_features: dictionary of morphological features. Should include
            pose_vectors of shape (3,3) and pose_fully_defined (bool).
        non_morphological_features: dictionary of non-morphological features.
        confidence: confidence in the state. In range [0,1].
        use_state: boolean indicating whether the state should be used or not.
        sender_id: string identifying the sender of the state.
        sender_type: string identifying the type of sender. Can be "SM" or "LM".
    """

    def __init__(
        self,
        location,
        morphological_features,
        non_morphological_features,
        confidence,
        use_state,
        sender_id,
        sender_type,
    ):
        """Initialize a state."""
        self.location = location
        # QUESTION: Divide into pose_dependent and pose_independent features instead?
        self.morphological_features = morphological_features
        self.non_morphological_features = non_morphological_features
        self.confidence = confidence
        self.use_state = use_state
        self.sender_id = sender_id
        self.sender_type = sender_type
        self._set_allowable_sender_types()
        if self.use_state:
            self._check_all_attributes()

    def __repr__(self):
        """Return a string representation of the object."""
        repr_string = (
            f"State from {self.sender_id}:\n"
            f"   Location: {np.round(self.location,3)}.\n"
            f"   Morphological Features: \n"
        )
        if self.morphological_features is not None:
            for feature in self.morphological_features:
                feat_val = self.morphological_features[feature]
                if type(feat_val) == np.ndarray:
                    feat_val = np.round(feat_val, 3)
                if feature == "pose_vectors":
                    repr_string += f"       {feature}: \n"
                    for vector in feat_val:
                        repr_string += f"           {vector}\n"
                else:
                    repr_string += f"       {feature}: {feat_val}\n"
        repr_string += f"   Non-Morphological Features: \n"
        if self.non_morphological_features is not None:
            for feature in self.non_morphological_features:
                feat_val = self.non_morphological_features[feature]
                if type(feat_val) in [np.ndarray, np.float64]:
                    feat_val = np.round(feat_val, 3)
                repr_string += f"       {feature}: {feat_val}\n"
        repr_string += (
            f"   Confidence: {self.confidence}\n"
            f"   Use State: {self.use_state}\n"
            f"   Sender Type: {self.sender_type}\n"
        )
        return repr_string

    def _set_allowable_sender_types(self):
        """Set the allowable sender types of this State class."""
        self.allowable_sender_types = ("SM", "LM")

    def transform_morphological_features(self, translation=None, rotation=None):
        """Apply translation and/or rotation to morphological features."""
        if translation is not None:
            self.location += translation
        if rotation is not None:
            self.morphological_features["pose_vectors"] = np.dot(
                rotation, self.morphological_features["pose_vectors"]
            )

    def set_displacement(self, displacement, ppf=None):
        """Add displacement (represented as dict) to state.

        TODO S: Add this to state or in another place?
        """
        self.displacement = {
            "displacement": displacement,
        }
        if ppf is not None:
            self.displacement["ppf"] = ppf

    def get_feature_by_name(self, feature_name):
        if feature_name in self.morphological_features.keys():
            feature_val = self.morphological_features[feature_name]
        elif feature_name in self.non_morphological_features.keys():
            feature_val = self.non_morphological_features[feature_name]
        else:
            raise ValueError(f"Feature {feature_name} not found in state.")
        return feature_val

    def get_nth_pose_vector(self, pose_vector_index):
        """Return the nth pose vector.

        When self.sender_type == "SM", the first pose vector is the point normal and the
        second and third are the curvature directions. When self.sender_type == "LM",
        the pose vectors correspond to the rotation of the object relative to the model
        learned of it.
        """
        return self.morphological_features["pose_vectors"][pose_vector_index]

    def get_point_normal(self):
        """Return the point normal vector.

        Raises:
            ValueError: If `self.sender_type` is not SM
        """
        if self.sender_type == "SM":
            return self.get_nth_pose_vector(0)
        else:
            raise ValueError("Sender type must be SM to get point normal.")

    def get_pose_vectors(self):
        """Return the pose vectors."""
        return self.morphological_features["pose_vectors"]

    def get_curvature_directions(self):
        """Return the curvature direction vectors.

        Raises:
            ValueError: If `self.sender_type` is not SM
        """
        if self.sender_type == "SM":
            return self.get_nth_pose_vector(1), self.get_nth_pose_vector(2)
        else:
            raise ValueError("Sender type must be SM to get curvature directions.")

    def get_on_object(self):
        """Return whether we think we are on the object or not.

        This is currently used in the policy to stay on the object.
        """
        if "on_object" in self.morphological_features.keys():
            return self.morphological_features["on_object"]
        else:
            # TODO: Use depth values to estimate on_object (either threshold or large
            # displacement)
            return True

    def _check_all_attributes(self):
        assert (
            "pose_vectors" in self.morphological_features.keys()
        ), "pose_vectors should be in morphological_features but keys are "
        f"{self.morphological_features.keys()}"
        # TODO S: may want to test length and angle between vectors as well
        assert self.morphological_features["pose_vectors"].shape == (
            3,
            3,
        ), "pose should be defined by three orthonormal unit vectors but pose_vectors "
        f"shape is {self.morphological_features['pose_vectors'].shape}"
        assert "pose_fully_defined" in self.morphological_features.keys()
        assert (
            type(self.morphological_features["pose_fully_defined"]) == bool
        ), "pose_fully_defined must be a boolean but type is "
        f"{type(self.morphological_features['pose_fully_defined'])}"
        assert self.location.shape == (
            3,
        ), f"Location must be a 3D vector but shape is {self.location.shape}"
        assert (
            self.confidence >= 0 and self.confidence <= 1
        ), f"Confidence must be in [0,1] but is {self.confidence}"
        assert (
            type(self.use_state) == bool
        ), f"use_state must be a boolean but is {type(self.use_state)}"
        assert (
            type(self.sender_id) == str
        ), f"sender_id must be string but is {type(self.sender_id)}"
        assert (
            self.sender_type in self.allowable_sender_types
        ), f"sender_type must be SM or LM but is\
            {self.sender_type}"


class GoalState(State):
    """Specialization of State for goal states with null (None) values allowed.

    Specialized form of state that still adheres to the cortical messaging protocol,
    but can have null (None) values associated with the location and morphological
    features.

    Used by goal-state generators (GSGs) to communicate goal states to other GSGs, and
    to motor actuators.

    The state variables generally have the same meanign as for the base State
    class, and they represent the target values for the receiving system. Thus
    if a goal-state specifies a particular object ID (non-morphological feature)
    in a particular pose (location and morphological features), then the receiving
    system should attempt to achieve that state.

    Note however that for the goal-state, the confidence corresponds to the conviction
        with which a GSG believes that the current goal-state should be acted upon.
        Float bound in [0,1.0].
    """

    def __init__(
        self,
        location,
        morphological_features,
        non_morphological_features,
        confidence,
        use_state,
        sender_id,
        sender_type,
        goal_tolerances,
    ):
        """Initialize a goal state.

        Args:
            location: ?
            morphological_features: ?
            non_morphological_features: ?
            confidence: ?
            use_state: ?
            sender_id: ?
            sender_type: ?
            goal_tolerances: Dictionary of tolerances that GSGs use when determining
                whether the current state of the LM matches the driving goal-state. As
                such, a GSG can send a goal state with more or less strict tolerances
                if certain elements of the state (e.g. the location of a mug vs its
                orientation) are more or less important.
        """
        self.goal_tolerances = goal_tolerances

        super().__init__(
            location,
            morphological_features,
            non_morphological_features,
            confidence,
            use_state,
            sender_id,
            sender_type,
        )

    def _set_allowable_sender_types(self):
        """Set the allowable sender types of this State class."""
        self.allowable_sender_types = "GSG"

    def _check_all_attributes(self):
        """Overwrite base attribute check to also allow for None values."""
        if self.morphological_features is not None:
            assert (
                "pose_vectors" in self.morphological_features.keys()
            ), "pose_vectors should be in morphological_features but keys are "
            f"{self.morphological_features.keys()}"
            assert np.any(
                self.morphological_features["pose_vectors"] == np.nan
            ) or self.morphological_features["pose_vectors"].shape == (
                3,
                3,
            ), f"pose should be undefined, or defined by three orthonormal unit vectors\
                but pose_vectors shape is\
                    {self.morphological_features['pose_vectors'].shape}"
            assert "pose_fully_defined" in self.morphological_features.keys()
            assert (
                type(self.morphological_features["pose_fully_defined"]) == bool
            ) or self.morphological_features[
                "pose_fully_defined"
            ] is None, "pose_fully_defined must be a boolean or None but type is "
            f"{type(self.morphological_features['pose_fully_defined'])}"
        if self.location is not None:
            assert self.location.shape == (
                3,
            ), f"Location must be a 3D vector but shape is {self.location.shape}"

        assert (
            self.confidence >= 0 and self.confidence <= 1
        ), f"Confidence must be in [0,1] but is {self.confidence}"
        assert (
            type(self.use_state) == bool
        ), f"use_state must be a boolean but is {type(self.use_state)}"
        assert (
            type(self.sender_id) == str
        ), f"sender_id must be string but is {type(self.sender_id)}"
        # Note *only* GSGs should create GoalState objects
        assert (
            self.sender_type in self.allowable_sender_types
        ), f"sender_type must be GSG but is {self.sender_type}"
