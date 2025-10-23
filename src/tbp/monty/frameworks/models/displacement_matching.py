# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import logging

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.environment_utils.graph_utils import get_edge_index
from tbp.monty.frameworks.models.graph_matching import GraphLM, GraphMemory
from tbp.monty.frameworks.models.object_model import GraphObjectModel
from tbp.monty.frameworks.utils.graph_matching_utils import is_in_ranges
from tbp.monty.frameworks.utils.sensor_processing import point_pair_features

logger = logging.getLogger(__name__)


class DisplacementGraphLM(GraphLM):
    """Learning module that uses displacement stored in graphs to recognize objects."""

    def __init__(
        self,
        k=None,
        match_attribute=None,
        tolerance=0.001,
        use_relative_len=False,
        graph_delta_thresholds=None,
    ):
        """Initialize Learning Module.

        Args:
            k: How many nearest neighbors should nodes in graphs connect to.
            match_attribute: Which displacement to use for matching.
                Should be in ['displacement', 'PPF'].
            tolerance: How close does an observed displacement have to be to a
                stored displacement to be considered a match,. defaults to 0.001
            use_relative_len: Whether to scale the displacements to achieve scale
                invariance. Only possible when using PPF.
            graph_delta_thresholds: Thresholds used to compare nodes in the graphs
                being learned, and thereby whether to include a new point or not. By
                default, we only consider the distance between points, using a threshold
                of 0.001 (determined in remove_close_points). Can also specify
                thresholds based on e.g. surface normal angle difference, or principal
                curvature magnitude difference.
        """
        super().__init__()
        self.graph_memory = DisplacementGraphMemory(
            graph_delta_thresholds=graph_delta_thresholds,
            k=k,
            match_attribute=match_attribute,
        )

        self.match_attribute = match_attribute
        self.tolerance = tolerance
        self.use_relative_len = use_relative_len

    # =============== Public Interface Functions ===============
    # ------------------- Main Algorithm -----------------------
    def reset(self):
        """Call this before each episode."""
        # reset possible matches for paths on objects
        (
            self.possible_matches,
            self.possible_paths,
            self.next_possible_paths,
            self.scale_factors,
        ) = self.graph_memory.get_initial_hypotheses()

    # ------------------ Getters & Setters ---------------------
    def get_unique_pose_if_available(self, object_id):
        """Compute (location, rotation, scale) of object.

        If we are sure about where on the object we are compare the sensed
        displacements to the observed displacements to calculate the pose, else
        return None.

        Returns:
            The pose and scale of the object.
        """
        pose_and_scale = None
        possible_paths = self.get_possible_paths()[object_id]

        # If multiple paths are possible, return None
        if len(possible_paths) == 1:
            # TODO H: Do we want to clean up all this first channel stuff
            # in the old LMs?
            first_channel = self.buffer.get_first_sensory_input_channel()
            detected_path = possible_paths[0]
            # get locations in model RF for nodes (int IDs) in the detected path
            detected_path_locs = self.graph_memory.get_locations_in_graph(
                object_id, input_channel=first_channel
            )[detected_path]
            # The location in object RF where the sensor is right now will be the last
            # on in the detected path
            current_model_loc = detected_path_locs[-1]
            model_displacements = np.array(
                [
                    np.array(detected_path_locs[i + 1] - detected_path_locs[i])
                    for i in range(len(detected_path_locs) - 1)
                ]
            )
            r_euler, _, r = self.get_object_rotation(
                sensed_displacements=np.array(
                    self.buffer.displacements[first_channel]["displacement"][1:]
                ),
                model_displacements=model_displacements,
                get_reverse_r=False,
            )
            # If r_euler is not None, we have a unique rotation
            if r_euler is not None:
                self.detected_rotation_r = r
                scale = self.get_object_scale(
                    np.array(
                        self.buffer.get_nth_displacement(1, input_channel=first_channel)
                    ),
                    model_displacements[0],
                )
                pose_and_scale = np.concatenate([current_model_loc, r_euler, [scale]])
                self.detected_pose = pose_and_scale
                lm_episode_stats = {
                    "detected_path": detected_path,
                    "detected_location_on_model": current_model_loc,
                    "detected_location_rel_body": self.buffer.get_current_location(
                        input_channel=first_channel
                    ),
                    "detected_rotation": r_euler,
                    "detected_rotation_quat": r.as_quat(),
                    "detected_scale": scale,
                }
                self.buffer.add_overall_stats(lm_episode_stats)
                logger.debug(f"(location, rotation, scale): {pose_and_scale}")

        return pose_and_scale

    def get_object_rotation(
        self, sensed_displacements, model_displacements, get_reverse_r=False
    ):
        """Calculate the rotation between two sets of displacement vectors.

        Args:
            sensed_displacements: The displacements that were sensed.
            model_displacements: The displacements in the model that were matched to the
                sensed displacements.
            get_reverse_r: Whether to get the rotation that turns the model such that it
                would produce the sensed_displacements (False) or the rotation needed to
                turn the sensed_displacements into the model displacements.

        Returns:
            The rotation in Euler angles, as a matrix, and as a Rotation object.
        """
        try:
            if get_reverse_r:
                r, msr = Rotation.align_vectors(
                    sensed_displacements, model_displacements
                )
            else:
                r, msr = Rotation.align_vectors(
                    model_displacements, sensed_displacements
                )
        except UserWarning:
            # This can happen if the displacements that were sampled lie in one plane
            # such that we can not determine the rotation along all three axes.
            print("could not determine rotation uniquely -> keep moving!")
            return None, None

        r_euler = np.round(r.as_euler("xyz", degrees=True), 3)
        r_matrix = r.as_matrix()
        return r_euler, r_matrix, r

    def get_object_scale(self, sensed_displacement, model_displacement):
        """Calculate the objects scale given sensed and model displacements.

        Returns:
            The scale of the object.
        """
        scale = np.linalg.norm(sensed_displacement) / np.linalg.norm(model_displacement)
        return scale

    # ------------------ Logging & Saving ----------------------

    # ======================= Private ==========================

    # ------------------- Main Algorithm -----------------------
    def _compute_possible_matches(self, observation, first_movement_detected=True):
        """Use the current observation to narrown down the possible matches.

        This is framed as a prediction problem. We take the current observation
        as a query and try to predict whether after the displacement we will still be
        on the object. In a next step we could also predict the feature that we sense
        next. The prediction is then compared with he actual observation (currently
        just whether we sensed on_object or not). If there is a prediction error, then
        we remove the object from the possible matches.

        Args:
            observation: The current observation.
            first_movement_detected: Whether the agent has moved yet. False on the first
                step.
        """
        if not first_movement_detected:
            return
        if self.match_attribute == "displacement":
            query = self.buffer.get_current_displacement(input_channel="first")
        elif self.match_attribute == "PPF":
            query = self.buffer.get_current_ppf(input_channel="first")
        else:
            logger.error("match_attribute not defined")

        # This is just whether we are on the object or not here.
        target = self._select_features_to_use(observation)

        if self.match_attribute == "PPF" and self.use_relative_len:
            query[0] = query[0] / self.buffer.get_first_displacement_len(
                input_channel="first"
            )

        logger.debug(f"query: {query}")

        self._update_possible_matches(query=query, target=target)

    def _update_possible_matches(self, query, target, threshold=0):
        """Update the list of possible matches.

        This is done by excluding objects that had a prediction
        error > threshold.

        Args:
            query: Incoming displacement.
            target: Whether we expect to be on the object of not after the given
                displacement.
            threshold: How high can the prediction error be for the object to still be
                considered? With binary predictions this is just 0. With features we may
                want to adapt this.

        """
        predictions = self._make_predictions(
            query=query,
            use_relative_len=self.use_relative_len,
        )
        prediction_error = self._get_prediction_error(predictions, target=target)

        for graph_id in prediction_error:
            if prediction_error[graph_id] > threshold:
                self.possible_matches.pop(graph_id)

    def _make_predictions(self, query, use_relative_len) -> dict:
        """Predict whether we will still be on the object given a displacement.

        Args:
            query: x,y,z query (where the sensor moved). For
                _predict_using_displacements this is a displacement vector.
            ranges: Range in which each element in the displacement can be to be
                classified as the same.
            use_relative_len: When matching, use relative displacement lengths instead
                of absolute. This may help with scale invariance.

        Returns:
            Binary predictions for each graph in possible_matches whether the object
            will be at the new location.

        """
        predictions = {}
        for graph_id in self.possible_matches:
            prediction = self._predict_using_displacements(
                np.array(query), graph_id, use_relative_len
            )
            predictions[graph_id] = prediction
        return predictions

    def _predict_using_displacements(
        self,
        displacement,
        graph_id,
        use_relative_len,
    ) -> int:
        """Predict whether we will still be on the object given a displacement.

        Takes a displacement as input (the last action that was performed) and checks
        for a specific object in memory whether the displacement could end up on one
        of its nodes given the current possible nodes after the past series of
        displacements.

        Args:
            displacement: 3D displacement vector of the last action. Will be compared to
                displacements between nodes of a graph.
            graph_id: id of graph which is used to make predictions.
            ranges: Range in which each element in the displacement can be to be
                classified as the same.
                -> how exact do they need to match?
            use_relative_len: When matching, use relative displacement lengths instead
                of absolute. This may help with scale invariance.

        Returns:
            Whether the displacement is on the object. 0 if not, 1 if it is.
        """
        # TODO: Due to the use of node IDs as paths start IDs it a bit tricky to use
        # multiple input channels & I am not sure if it is worth the time investment atm
        # since we don't actively use this LM. So for now we just take the first input
        # channel here.
        first_input_channel = list(self.possible_matches[graph_id].keys())[0]
        displacement_plus_tolerance = np.stack(
            [displacement - self.tolerance, displacement + self.tolerance],
            axis=1,
        )
        self.possible_paths[graph_id] = self.next_possible_paths[graph_id]
        # possible_next_nodes = []
        new_possible_paths = []
        current_possible_paths = []
        path_scale_factors = []
        # for node in self.possible_nodes[graph_id]:
        for path_id, path in enumerate(self.possible_paths[graph_id]):
            previous_node = path[-2]
            current_node = path[-1]

            edge_id = get_edge_index(
                self.possible_matches[graph_id][first_input_channel],
                previous_node,
                current_node,
            )
            node_displacement = (
                self.possible_matches[graph_id][first_input_channel]
                .edge_attr[edge_id]
                .detach()
                .clone()
            )

            if use_relative_len:
                node_displacement[0] = (
                    node_displacement[0] / self.scale_factors[graph_id][path_id]
                )

            if is_in_ranges(node_displacement, displacement_plus_tolerance):
                current_possible_paths.append(path)
                edges_of_node = np.where(
                    self.possible_matches[graph_id][first_input_channel].edge_index[0]
                    == current_node
                )[0]
                next_nodes = self.possible_matches[graph_id][
                    first_input_channel
                ].edge_index[1][edges_of_node]

                for next_node in next_nodes:
                    new_possible_paths.append(np.append(path, int(next_node)))
                    path_scale_factors.append(self.scale_factors[graph_id][path_id])

        self.possible_paths[graph_id] = current_possible_paths
        self.next_possible_paths[graph_id] = new_possible_paths
        self.scale_factors[graph_id] = path_scale_factors
        # logger.info(
        #     "possible paths for "
        #     + graph_id
        #     + ": "
        #     + str(self.possible_paths[graph_id])
        # )
        # logger.info(
        #     "next possible paths for "
        #     + graph_id
        #     + ": "
        #     + str(self.next_possible_paths[graph_id])
        # )

        if len(self.possible_paths[graph_id]) == 0:
            return 0
        else:
            return 1

    def _get_prediction_error(self, predictions, target):
        """Calculate the prediction error (binary if not using features).

        Args:
            predictions: A binary prediction on the objects morphology (object
                there or not) per graph.
            target: The actual sensation at the new location (also binary)

        Returns:
            Binary prediction error for each graph: int(target != prediction)
        """
        prediction_error = {}
        for graph_id in predictions:
            prediction_error[graph_id] = int(target != predictions[graph_id])
        return prediction_error

    # ------------------------ Helper --------------------------
    def _add_displacements(self, obs):
        """Add displacements to the current observation.

        The observation consists of features at a location. To get the displacement we
        have to look at the previous observation stored in the buffer.

        TODO: Should we move this and a (short term) buffer to the sensor module?

        Returns:
            The observations with displacements added.
        """
        displacement = np.zeros(3)
        ppf = np.zeros(4)
        # TODO S: calculate displacements for each separately (mostly for rotation disp)
        obs_to_use = obs[0]

        if len(self.buffer) > 0:
            # TODO S: Make sure result of get_current_location() and get_current_pose()
            # is on object (should always be atm).
            displacement = np.array(
                obs_to_use.location
            ) - self.buffer.get_current_location(input_channel=obs_to_use.sender_id)

            pos1 = torch.tensor(
                self.buffer.get_current_location(input_channel=obs_to_use.sender_id)
            )
            pos2 = torch.tensor(obs_to_use.location)
            norm1 = torch.tensor(
                # element 0 of current pose is location, element 1 is surface normal
                self.buffer.get_current_pose(input_channel=obs_to_use.sender_id)[1],
                dtype=torch.float64,
            )
            norm2 = torch.tensor(
                obs_to_use.get_nth_pose_vector(pose_vector_index=0),
                dtype=torch.float64,
            )
            ppf = point_pair_features(pos1, pos2, norm1, norm2)
        for o in obs:
            o.set_displacement(displacement=displacement, ppf=ppf)
        return obs

    def _select_features_to_use(self, states) -> int:
        """Extract on_object from observed features to use as target.

        Returns:
            Whether we are on the object or not as integer.
        """
        morph_features = states[0].morphological_features
        # TODO S: decide if we want to store on_object in state
        if "on_object" in morph_features:
            on_object = morph_features["on_object"]
        else:
            on_object = 1
        return int(on_object)

    # ----------------------- Logging --------------------------
    def _add_detailed_stats(self, stats):
        stats["possible_paths"] = self.get_possible_paths()
        return stats


class DisplacementGraphMemory(GraphMemory):
    """Graph memory that stores graphs with displacements as edges."""

    def __init__(self, match_attribute, *args, **kwargs):
        """Initialize Graph memory."""
        super().__init__(*args, **kwargs)
        self.match_attribute = match_attribute

    # =============== Public Interface Functions ===============

    # ------------------- Main Algorithm -----------------------
    def get_initial_hypotheses(self):
        possible_matches = self.get_all_models_in_memory()
        possible_paths = {}
        next_possible_paths = {}  # Need this for scale factors to work
        scale_factors = {}

        for graph_id in self.get_memory_ids():
            first_input_channel = self.get_input_channels_in_graph(graph_id)[0]
            next_possible_paths[graph_id] = np.swapaxes(
                self.get_graph(graph_id, first_input_channel).edge_index, 0, 1
            )
            if self.get_graph(graph_id, first_input_channel).x.dim() > 1:
                # Features of nodes contain more than just IDs (i.e. RGBA)
                possible_paths[graph_id] = self.get_graph_node_ids(
                    graph_id, first_input_channel
                )
            else:
                possible_paths[graph_id] = np.array(
                    self.get_graph(graph_id, first_input_channel).x
                )
            scale_factors[graph_id] = np.array(
                self.get_graph(graph_id, first_input_channel).edge_attr[:, 0]
            )
        return (
            possible_matches,
            possible_paths,
            next_possible_paths,
            scale_factors,
        )

    # ------------------ Logging & Saving ----------------------
    def load_state_dict(self, state_dict):
        """Load graphs into memory from a state_dict and add point pair features."""
        logger.info("loading models")
        for obj_name, model in state_dict.items():
            logger.debug(f"loading {obj_name}: {model}")
            for input_channel in model:
                if (self.match_attribute == "PPF") and (
                    model[input_channel].has_ppf is False
                ):
                    model[input_channel].add_ppf_to_graph()
            self._add_graph_to_memory(model, obj_name)

    # ======================= Private ==========================

    # ------------------- Main Algorithm -----------------------
    def _build_graph(self, locations, features, graph_id, input_channel):
        """Build a k nearest neighbor graph from a list of observations.

        Custom version of super._build_graph that adds edges to the graph
        and attaches point pair features to them if this is the match_attribute.

        Args:
            locations: List of x,y,z locations.
            features: Features observed at the locations.
            graph_id: Name of the object.
            input_channel: ?
        """
        logger.info(f"Adding a new graph to memory.")

        model = GraphObjectModel(
            object_id=graph_id,
        )
        graph_delta_thresholds = (
            None
            if self.graph_delta_thresholds is None
            else self.graph_delta_thresholds[input_channel]
        )
        model.build_model(
            locations,
            features,
            k_n=self.k,
            graph_delta_thresholds=graph_delta_thresholds,
        )
        if self.match_attribute == "PPF":
            model.add_ppf_to_graph()
        if graph_id not in self.models_in_memory:
            self.models_in_memory[graph_id] = {}
        self.models_in_memory[graph_id][input_channel] = model
        logger.info(f"Added new graph with id {graph_id} to memory.")
