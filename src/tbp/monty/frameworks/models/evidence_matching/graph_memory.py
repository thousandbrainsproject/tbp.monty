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

from tbp.monty.frameworks.models.graph_matching import GraphMemory
from tbp.monty.frameworks.models.object_model import (
    GraphObjectModel,
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
        """Get rotation features from all N nodes. shape=(N, 3, 3).

        Returns:
            The rotation features from all N nodes. shape=(N, 3, 3).
        """
        all_node_r_features = self.get_features_at_node(
            graph_id,
            input_channel,
            self.get_graph_node_ids(graph_id, input_channel),
            feature_keys=["pose_vectors"],
        )
        node_directions = all_node_r_features["pose_vectors"]
        num_nodes = len(node_directions)
        node_directions = node_directions.reshape((num_nodes, 3, 3))
        return node_directions

    # ======================= Private ==========================

    # ------------------- Main Algorithm -----------------------
    def _add_graph_to_memory(self, model, graph_id):
        """Add pretrained graph to memory.

        Initializes GridObjectModel and calls set_graph.

        Args:
            model: new model to be added to memory
            graph_id: id of graph that should be added

        """
        self.models_in_memory[graph_id] = {}
        for input_channel in model.keys():
            channel_model = model[input_channel]
            try:
                if isinstance(channel_model, GraphObjectModel):
                    # When loading a model trained with a different LM, need to convert
                    # it to the GridObjectModel (with use_original_graph == True)
                    loaded_graph = channel_model._graph
                    channel_model = self._initialize_model_with_graph(
                        graph_id, loaded_graph
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
        """Build a graph from a list of features at locations and add to memory.

        This initializes a new GridObjectModel and calls model.build_graph.

        Args:
            locations: List of x,y,z locations.
            features: List of features.
            graph_id: name of new graph.
            input_channel: ?
        """
        logger.info(f"Adding a new graph to memory.")

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
        object_scale,
    ):
        """Add new observations into an existing graph.

        Args:
            locations: List of x,y,z locations.
            features: Features observed at the locations.
            graph_id: ID of the existing graph.
            input_channel: ?
            object_location_rel_body: location of the sensor in body reference frame
            location_rel_model: location of sensor in model reference frame
            object_rotation: rotation of the sensed object relative to the model
            object_scale: scale of the object relative to the model of it
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
