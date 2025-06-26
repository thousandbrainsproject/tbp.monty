# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import logging
import threading
from typing import Protocol

from tbp.monty.frameworks.models.abstract_monty_classes import LearningModule
from tbp.monty.frameworks.models.buffer import FeatureAtLocationBuffer

logger = logging.getLogger(__name__)


class MatchComputer(Protocol):
    def compute_possible_matches(
        self, observations, first_movement_detected=True
    ) -> None: ...


class NotImplementedMatchComputer:
    def compute_possible_matches(
        self, instance: LearningModule, observations, first_movement_detected=True
    ):
        raise NotImplementedError("Need to implement way to update memory hypotheses")


class EvidenceMatchComputer:
    def __init__(
        self,
        buffer: FeatureAtLocationBuffer,
        tolerances: dict,
    ):
        self.buffer = buffer
        self.tolerances = tolerances

    def compute_possible_matches(
        self, instance: LearningModule, observations, first_movement_detected=True
    ):
        """Compute possible matches for the given observations.

        Args:
            instance: The instance of the GraphLM class.
            observations: The observations to use for computing possible matches.
            first_movement_detected: Whether the agent has moved since the buffer reset
                signal.
        """
        if first_movement_detected:
            query = [
                self._select_features(observations),
                self.buffer.get_current_displacement(input_channel="all"),
            ]
        else:
            query = [
                self._select_features(observations),
                None,
            ]

        logger.debug(f"query: {query}")

        self._update_possible_matches(instance, query=query)

    def _select_features(self, observations) -> dict:
        """Extract the features from observations that are specified in tolerances.

        Returns:
            dict: Features to use.
        """
        features_to_use = {}
        for observation in observations:
            input_channel = observation.sender_id
            features_to_use[input_channel] = {}
            for feature in observation.morphological_features.keys():
                # in evidence matching pose_vectors are always added to tolerances
                # since they are requires for matching.
                if (
                    feature in self.tolerances[input_channel].keys()
                    or feature == "pose_fully_defined"
                ):
                    features_to_use[input_channel][feature] = (
                        observation.morphological_features[feature]
                    )
            for feature in observation.non_morphological_features.keys():
                if feature in self.tolerances[input_channel].keys():
                    features_to_use[input_channel][feature] = (
                        observation.non_morphological_features[feature]
                    )

        return features_to_use

    def _update_possible_matches(self, instance: LearningModule, query):
        """Update evidence for each hypothesis instead of removing them."""
        thread_list = []
        for graph_id in instance.get_all_known_object_ids():
            if instance.use_multithreading:
                # assign separate thread on same CPU to each objects update.
                # Since the updates of different objects are independent of
                # each other we can do this.
                t = threading.Thread(
                    target=self._update_evidence,
                    args=(query[0], query[1], graph_id),
                )
                thread_list.append(t)
            else:  # This can be useful for debugging.
                self._update_evidence(query[0], query[1], graph_id)
        if instance.use_multithreading:
            # TODO: deal with keyboard interrupt
            for thread in thread_list:
                # start executing _update_evidence in each thread.
                thread.start()
            for thread in thread_list:
                # call this to prevent main thread from continuing in code
                # before all evidences are updated.
                thread.join()
        # NOTE: would not need to do this if we are still voting
        # Call this update in the step method?
        instance.possible_matches = self._threshold_possible_matches()
        instance.current_mlh = self._calculate_most_likely_hypothesis()
