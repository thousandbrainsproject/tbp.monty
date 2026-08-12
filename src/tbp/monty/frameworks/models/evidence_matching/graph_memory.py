# Copyright 2025-2026 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import copy
import logging

import numpy as np

from tbp.monty.frameworks.models.graph_matching import GraphMemory
from tbp.monty.frameworks.models.object_model import (
    GridObjectModel,
    GridTooSmallError,
)

logger = logging.getLogger(__name__)


class EvidenceGraphMemory(GraphMemory):
    """Custom GraphMemory that stores GridObjectModel instead of GraphObjectModel."""

    def __init__(
        self,
        max_nodes_per_graph,
        max_graph_size,
        num_model_voxels_per_dim,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.max_nodes_per_graph = max_nodes_per_graph
        self.max_graph_size = max_graph_size
        self.num_model_voxels_per_dim = num_model_voxels_per_dim

    # =============== Public Interface Functions ===============

    # ------------------- Main Algorithm -----------------------

    # ------------------ Getters & Setters ---------------------
    def get_initial_hypotheses(self):
        return self.get_memory_ids()

    def get_rotation_features_at_all_nodes(self, graph_id, input_channel):
        """Get rotation features from all N nodes. Shape=(N, 3, 3).

        Returns:
            The rotation features from all N nodes. Shape=(N, 3, 3).
        """
        all_node_r_features = self.get_features_at_node(
            graph_id,
            input_channel,
            self.get_graph_node_ids(graph_id, input_channel),
            feature_keys=["pose_vectors"],
        )
        node_directions = all_node_r_features["pose_vectors"]
        num_nodes = len(node_directions)
        return node_directions.reshape((num_nodes, 3, 3))

    def get_features_by_name(
        self, graph_id: str, input_channel: str
    ) -> dict[str, np.ndarray]:
        """Return a graph's stored features as a `{feature_name: (N, d)}` dict.

        `node_ids` is excluded since the grid build path
        derives node ids itself.

        Args:
            graph_id: ID of the graph to read features from.
            input_channel: Identifier of the input channel.

        Returns:
            Features keyed by name, each an `(N, d)` array.
        """
        model = self.get_graph(graph_id, input_channel)
        x = np.asarray(model.x)
        return {
            feature: x[:, ids[0] : ids[1]]
            for feature, ids in model.feature_mapping.items()
            if feature != "node_ids"
        }

    # ======================= Private ==========================

    # ------------------- Main Algorithm -----------------------
    def _add_graph_to_memory(self, model, graph_id):
        """Add a pretrained graph to memory.

        Initializes GridObjectModel and calls set_graph.

        Args:
            model: New model to be added to memory.
            graph_id: ID of the graph that should be added.

        """
        self.models_in_memory[graph_id] = {}
        for input_channel in model:
            channel_model = model[input_channel]
            try:
                if not isinstance(channel_model, GridObjectModel):
                    # When loading a model trained with a different LM, need to convert
                    # it to the GridObjectModel (with use_original_graph == True)
                    loaded_graph = channel_model._graph
                    channel_model = self._initialize_model_with_graph(
                        graph_id, loaded_graph
                    )
                else:
                    # serialization seems to mess up the sparse tensors, so we need to
                    # coalesce them again.
                    if channel_model._observation_count is not None:
                        channel_model._observation_count = (
                            channel_model._observation_count.coalesce()
                        )
                    if channel_model._feature_grid is not None:
                        channel_model._feature_grid = (
                            channel_model._feature_grid.coalesce()
                        )
                    if channel_model._location_grid is not None:
                        channel_model._location_grid = (
                            channel_model._location_grid.coalesce()
                        )

                logger.info(f"Loaded {model} for {input_channel}")
                self.models_in_memory[graph_id][input_channel] = channel_model
            except GridTooSmallError:
                logger.info("Grid too small for given locations. Not adding to memory.")

    def _initialize_model_with_graph(self, graph_id, graph):
        model = GridObjectModel(
            object_id=graph_id,
            max_nodes=self.max_nodes_per_graph,
            max_size=self.max_graph_size,
            num_voxels_per_dim=self.num_model_voxels_per_dim,
        )
        # Keep benchmark results constant by still using original graph for
        # matching when loading pretrained models.
        model.use_original_graph = True
        model.set_graph(graph)
        return model

    def _build_graph(self, locations, features, graph_id, input_channel):
        """Build a graph from a list of features at locations and add it to memory.

        This initializes a new GridObjectModel and calls model.build_graph.

        Args:
            locations: List of x, y, z locations.
            features: List of features.
            graph_id: ID of the new graph.
            input_channel: Identifier of the input channel.
        """
        logger.info("Adding a new graph to memory.")

        model = GridObjectModel(
            object_id=graph_id,
            max_nodes=self.max_nodes_per_graph,
            max_size=self.max_graph_size,
            num_voxels_per_dim=self.num_model_voxels_per_dim,
        )
        try:
            model.build_model(locations=locations, features=features)

            if graph_id not in self.models_in_memory:
                self.models_in_memory[graph_id] = {}
            self.models_in_memory[graph_id][input_channel] = model

            logger.info(f"Added new graph with id {graph_id} to memory.")
            logger.info(model)
        except GridTooSmallError:
            logger.info(
                "Grid too small for given locations. Not building a model "
                f"for {graph_id}"
            )

    def _extend_graph(
        self,
        locations,
        features,
        graph_id,
        input_channel,
        object_location_rel_body,
        location_rel_model,
        object_rotation,
    ):
        """Add new observations into an existing graph.

        Args:
            locations: List of x, y, z locations.
            features: Features observed at the provided locations.
            graph_id: ID of the existing graph.
            input_channel: Identifier of the input channel.
            object_location_rel_body: Location of the sensor in the body reference
                frame.
            location_rel_model: Location of the sensor in the model reference frame.
            object_rotation: Rotation of the sensed object relative to the model.
        """
        logger.info(f"Updating existing graph for {graph_id}")

        try:
            self.models_in_memory[graph_id][input_channel].update_model(
                locations=locations,
                features=features,
                location_rel_model=location_rel_model,
                object_location_rel_body=object_location_rel_body,
                object_rotation=object_rotation,
            )
            logger.info(
                f"Extended graph {graph_id} with new points. New model:\n"
                f"{self.models_in_memory[graph_id]}"
            )
        except GridTooSmallError:
            logger.info("Grid too small for given locations. Not updating model.")

    def _init_merged_model(
        self, first_model: GridObjectModel, first_graph_id: str, channel: str
    ) -> GridObjectModel:
        """Create the base model that other objects' points get merged into.

        Models grown during learning already have a grid, so a deep copy is
        extended directly and keeps its location mapping and observation counts.
        Models loaded from a pretrained graph keep using their original graph and
        never built a grid, so `update_model` cannot extend them; a fresh grid is
        built from the model's own nodes first (observation counts start at 1).

        Args:
            first_model: The first graph's model for this channel.
            first_graph_id: ID of the first graph (to read its features).
            channel: Identifier of the input channel.

        Returns:
            A `GridObjectModel` with a grid, ready to be extended via
            `update_model`. The caller sets its final `object_id`.
        """
        if getattr(first_model, "_location_scale_factor", None) is not None:
            return copy.deepcopy(first_model)

        model = GridObjectModel(
            object_id=first_graph_id,
            max_nodes=self.max_nodes_per_graph,
            max_size=self.max_graph_size,
            num_voxels_per_dim=self.num_model_voxels_per_dim,
        )
        model.build_model(
            locations=np.asarray(first_model.pos),
            features=self.get_features_by_name(first_graph_id, channel),
        )
        return model

    def _merge_graphs(
        self,
        first_graph_id: str,
        merge_data: dict[str, list[tuple]],
        new_graph_id: str,
        old_graph_ids: list[str],
        location_rel_model: np.ndarray,
    ) -> bool:
        """Merge source graphs into a copy of the first graph's model.

        For each channel, the first graph's model is extended via `update_model`
        with every other object's points. `update_model` applies the
        reference-frame transform and rotates `pose_vectors`. Source graphs are
        only removed once every channel has succeeded. Since a copy is mutated, a
        `GridTooSmallError` leaves memory unchanged.

        Args:
            first_graph_id: ID of the graph whose model/grid is extended.
            merge_data: Per-channel list of
                `(locations, features, object_rotation, object_location_rel_body)`
                tuples, one per non-first source object.
            new_graph_id: ID to register the merged model under.
            old_graph_ids: IDs of the source graphs to remove from memory.
            location_rel_model: Location of the reference point in the first
                model's reference frame (the first object's MLH location).

        Returns:
            Whether the merge succeeded. On failure memory is left unchanged.
        """
        logger.info(f"Merging graphs {old_graph_ids} into new graph {new_graph_id}.")

        merged_models = {}
        for channel, entries in merge_data.items():
            first_model = self.models_in_memory[first_graph_id][channel]
            try:
                model = self._init_merged_model(first_model, first_graph_id, channel)
                model.object_id = new_graph_id
                for (
                    locations,
                    features,
                    object_rotation,
                    object_location_rel_body,
                ) in entries:
                    model.update_model(
                        locations=locations,
                        features=features,
                        location_rel_model=location_rel_model,
                        object_location_rel_body=object_location_rel_body,
                        object_rotation=object_rotation,
                    )
            except GridTooSmallError:
                logger.info(
                    f"Merged points for {channel} do not fit in "
                    f"{first_graph_id}'s grid. Aborting merge, memory unchanged."
                )
                return False
            merged_models[channel] = model

        self.models_in_memory[new_graph_id] = merged_models
        for old_graph_id in old_graph_ids:
            self.remove_graph_from_memory(old_graph_id)
            logger.info(f"Removed graph {old_graph_id} from memory.")
        return True
