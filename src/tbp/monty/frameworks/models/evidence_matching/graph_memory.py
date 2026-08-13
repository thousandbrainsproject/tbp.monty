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
from tbp.monty.frameworks.utils.spatial_arithmetics import (
    apply_rf_transform_to_points,
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

        # Optional visualization hook. Teleop can install a synchronous callback here.
        # Monty/Janus itself does not depend on Teleop.
        self._merge_animation_callback = None

    # =============== Public Interface Functions ===============

    # ------------------- Main Algorithm -----------------------

    # ------------------ Getters & Setters ---------------------
    def get_initial_hypotheses(self):
        return self.get_memory_ids()

    def set_merge_animation_callback(self, callback) -> None:
        """Set an optional callback invoked immediately before a successful merge commits.

        The callback is visualization-only. Exceptions raised by it are caught by
        `_merge_graphs` so visualization can never prevent a memory merge.
        """
        self._merge_animation_callback = callback

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

    def _merge_graphs(
        self,
        first_graph_id: str,
        merge_data: dict[str, list[tuple]],
        new_graph_id: str,
        old_graph_ids: list[str],
        location_rel_model: np.ndarray,
    ) -> bool:
        """Merge source graphs into a new model built on a fresh grid.

        For each channel, every other object's points are transformed into the
        first graph's reference frame and combined with the first graph's own
        points. A fresh model is then built with the default grid parameters,
        anchored so the combined point cloud's bounding-box center sits at the
        center of the grid. Source graphs are only removed once every channel
        has succeeded, so a `GridTooSmallError` leaves memory unchanged.

        If a merge-animation callback is installed, it receives the exact
        reference-frame transform inputs and final merged point cloud immediately
        before a successful merge is committed to memory.

        Args:
            first_graph_id: ID of the graph whose reference frame is used.
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

        source_graph_ids = [
            graph_id
            for graph_id in old_graph_ids
            if graph_id != first_graph_id
        ]

        merged_models = {}
        animation_channels = {}

        for channel, entries in merge_data.items():
            first_locations = np.asarray(
                self.get_locations_in_graph(first_graph_id, channel)
            )
            locations = [first_locations]
            first_features = self.get_features_by_name(
                first_graph_id,
                channel,
            )
            features = [first_features]

            animate_channel = len(entries) == len(source_graph_ids)
            if not animate_channel:
                logger.warning(
                    "Cannot associate merge-animation entries with source graph IDs: "
                    "%s entries for %s source graphs.",
                    len(entries),
                    len(source_graph_ids),
                )

            animation_entries = []

            for entry_index, (
                other_locations,
                other_features,
                object_rotation,
                object_location_rel_body,
            ) in enumerate(entries):
                transformed_locations, transformed_features = (
                    apply_rf_transform_to_points(
                        locations=other_locations,
                        features=other_features,
                        location_rel_model=location_rel_model,
                        object_location_rel_body=object_location_rel_body,
                        object_rotation=object_rotation,
                    )
                )
                locations.append(transformed_locations)
                features.append(transformed_features)

                if animate_channel:
                    animation_entries.append(
                        {
                            "source_graph_id": source_graph_ids[entry_index],
                            "locations": np.asarray(
                                other_locations,
                                dtype=float,
                            ).copy(),
                            "features": copy.deepcopy(other_features),
                            "object_rotation": copy.deepcopy(object_rotation),
                            "object_location_rel_body": np.asarray(
                                object_location_rel_body,
                                dtype=float,
                            ).copy(),
                        }
                    )

            combined_locations = np.vstack(locations)
            combined_features = {
                name: np.concatenate([f[name] for f in features])
                for name in features[0]
            }
            grid_center = (
                combined_locations.min(axis=0) + combined_locations.max(axis=0)
            ) / 2

            model = GridObjectModel(
                object_id=new_graph_id,
                max_nodes=self.max_nodes_per_graph,
                max_size=self.max_graph_size,
                num_voxels_per_dim=self.num_model_voxels_per_dim,
            )
            try:
                model.build_model(
                    locations=combined_locations,
                    features=combined_features,
                    start_location=grid_center,
                )
            except GridTooSmallError:
                logger.info(
                    f"Merged points for {channel} span more than the grid's "
                    "max_size. Aborting merge, memory unchanged."
                )
                return False

            merged_models[channel] = model

            if animate_channel:
                animation_channels[channel] = {
                    "target_points": np.asarray(
                        first_locations,
                        dtype=float,
                    ).copy(),

                    "target_hsv": (
                        np.asarray(
                            first_features["hsv"],
                            dtype=float,
                        ).copy()
                        if "hsv" in first_features
                        else None
                    ),

                    "entries": animation_entries,

                    "merged_points": np.asarray(
                        model.pos,
                        dtype=float,
                    ).copy(),

                    "merged_hsv": (
                        np.asarray(
                            model.get_values_for_feature("hsv"),
                            dtype=float,
                        ).copy()
                        if "hsv" in model.feature_mapping
                        else None
                    ),
                }

        # At this point every temporary merged model was built successfully,
        # but models_in_memory still contains the original unmerged objects.
        if (
            self._merge_animation_callback is not None
            and animation_channels
        ):
            try:
                self._merge_animation_callback(
                    memory=self,
                    first_graph_id=first_graph_id,
                    new_graph_id=new_graph_id,
                    old_graph_ids=tuple(old_graph_ids),
                    location_rel_model=np.asarray(
                        location_rel_model,
                        dtype=float,
                    ).copy(),
                    channels=animation_channels,
                )
            except Exception:
                logger.exception(
                    "Merge-animation callback failed; "
                    "continuing with graph merge."
                )

        self.models_in_memory[new_graph_id] = merged_models
        for old_graph_id in old_graph_ids:
            self.remove_graph_from_memory(old_graph_id)
            logger.info(f"Removed graph {old_graph_id} from memory.")
        return True
