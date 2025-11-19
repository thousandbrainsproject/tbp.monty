# Copyright 2025 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import copy
import logging

import numpy as np
import torch
import torch_geometric
import torch_geometric.transforms as T
from scipy.spatial import KDTree
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data

from tbp.monty.frameworks.models.abstract_monty_classes import ObjectModel
from tbp.monty.frameworks.utils.graph_matching_utils import get_correct_k_n
from tbp.monty.frameworks.utils.object_model_utils import (
    NumpyGraph,
    build_point_cloud_graph,
    circular_mean,
    expand_index_dims,
    get_most_common_bool,
    get_most_common_value,
    get_values_from_dense_last_dim,
    increment_sparse_tensor_by_count,
    pose_vector_mean,
    remove_close_points,
    torch_graph_to_numpy,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import apply_rf_transform_to_points

logger = logging.getLogger(__name__)


class GraphObjectModel(ObjectModel):
    """Object model class that represents object as graphs."""

    def __init__(self, object_id):
        """Initialize the object model.

        Args:
            object_id: id of the object
        """
        logger.info(f"init object model with id {object_id}")
        self.object_id = object_id
        self._graph = None
        self.has_ppf = False

    # =============== Public Interface Functions ===============
    # ------------------- Main Algorithm -----------------------
    def build_model(self, locations, features, k_n, graph_delta_thresholds):
        """Build graph from locations and features sorted into grids."""
        graph = self._build_adjacency_graph(
            locations,
            features,
            k_n=k_n,
            graph_delta_thresholds=graph_delta_thresholds,
            old_graph_index=0,
        )
        self.k_n = k_n
        self.graph_delta_thresholds = graph_delta_thresholds
        self.set_graph(graph)
        logger.info(f"built graph {self._graph}")

    def update_model(
        self,
        locations,
        features,
        location_rel_model,
        object_location_rel_body,
        object_rotation,
    ):
        """Add new locations and features into grids and rebuild graph."""
        rf_locations, rf_features = apply_rf_transform_to_points(
            locations=locations,
            features=features,
            location_rel_model=location_rel_model,
            object_location_rel_body=object_location_rel_body,
            object_rotation=object_rotation,
        )

        all_locations, all_features = self._combine_graph_information(
            rf_locations,
            rf_features,
        )

        logger.info("building graph")
        new_graph = self._build_adjacency_graph(
            all_locations,
            all_features,
            k_n=self.k_n,
            graph_delta_thresholds=self.graph_delta_thresholds,
            old_graph_index=self.num_nodes,
        )
        self.set_graph(new_graph)
        if self.has_ppf:
            self.add_ppf_to_graph()

    def add_ppf_to_graph(self):
        """Add point pair features to graph edges."""
        self._graph = T.PointPairFeatures(cat=False)(self._graph)
        self.has_ppf = True

    # ------------------ Getters & Setters ---------------------
    # Keep original properties of graphs for backward compatibility.
    @property
    def x(self):
        # TODO: How do we want to deal with self._graph being None?
        if self._graph is not None:
            return self._graph.x

    @property
    def pos(self):
        if self._graph is not None:
            return self._graph.pos

    @property
    def norm(self):
        if self._graph is not None:
            return self._graph.norm

    @property
    def feature_mapping(self):
        if self._graph is not None:
            return self._graph.feature_mapping

    @property
    def edge_index(self):
        if (self._graph is not None) and ("edge_index" in self._graph.keys):
            return self._graph.edge_index

    @property
    def edge_attr(self):
        if (self._graph is not None) and ("edge_attr" in self._graph.keys):
            return self._graph.edge_attr

    @property
    def num_nodes(self):
        return len(self._graph.pos)

    @property
    def feature_ids_in_graph(self):
        if self._graph is not None:
            return self._graph.feature_mapping.keys()

    def set_graph(self, graph):
        """Set self._graph property with given graph (i.e. from pretraining)."""
        self._graph = graph

    def get_values_for_feature(self, feature):
        featue_idx = self.feature_mapping[feature]
        return self.x[:, featue_idx[0] : featue_idx[1]]

    # ------------------ Logging & Saving ----------------------
    def __repr__(self):
        """Return a string representation of the object."""
        if self._graph is not None:
            return self._graph.__repr__()

        return f"Model for {self.object_id}:\n   No graph stored yet."

    # ======================= Private ==========================
    # ------------------- Main Algorithm -----------------------

    def _combine_graph_information(
        self,
        locations,
        features,
    ):
        """Combine new observations with those already stored in a graph.

        Combines datapoints from an existing graph and new points collected in the
        buffer using the detected pose. This is a util function for extend_graph.

        Args:
            locations: new observed locations (x,y,z)
            features: new observed features (dict)

        Returns:
            Combines features at locations with new locations transformed into the
            graph's reference frame.
        """
        old_points = self.pos
        feature_mapping = self.feature_mapping

        all_features = {}

        # Iterate through the different feature types, stacking on (i.e. appending)
        # those features associated w/ candidate new points to the old-graph point
        # features
        for feature in features.keys():
            new_feat = np.array(features[feature])
            if len(new_feat.shape) == 1:
                new_feat = new_feat.reshape((new_feat.shape[0], 1))
            if feature in feature_mapping.keys():
                feature_idx = feature_mapping[feature]
                old_feat = np.array(self.x)[:, feature_idx[0] : feature_idx[1]]
            else:
                # NOTE: currently never happens since all features are present at every
                # step. Should we remove this? Will this ever happen in the future?
                # add new feature into graph
                feature_start_idx = self.x.shape[-1]
                feature_len = new_feat.shape[-1]
                feature_mapping[feature] = [
                    feature_start_idx,
                    feature_start_idx + feature_len,
                ]
                # Pad with zeros for existing locations in graph
                old_feat = np.zeros((old_points.shape[0], feature_len))

            both_feat = np.vstack([old_feat, new_feat])
            all_features[feature] = both_feat

            for graph_feature in self.feature_ids_in_graph:
                if graph_feature not in features.keys() and graph_feature != "node_ids":
                    raise NotImplementedError(
                        f"{graph_feature} is represented in graph but",
                        " was not observed at this step. Implement padding with nan.",
                    )

        all_locations = np.vstack([old_points, locations])

        return all_locations, all_features

    def _build_adjacency_graph(
        self, locations, features, k_n, graph_delta_thresholds, old_graph_index
    ):
        """Build graph from observations with nodes linking to the n nearest neighbors.

        if k_n == None, this function will just return a graph without edges.

        Args:
            locations: array of x, y, z positions in space
            features: dictionary of features at locations
            k_n: How many nearest nodes each node should link to. If None,
                just return a point cloud with no links
            graph_delta_thresholds: dictionary of thresholds; if the L-2 distance
                between the locations of two observations (or other feature-distance
                measure) is below all of the given thresholds, then a point will be
                considered insufficiently interesting to be added.
            old_graph_index: If the graph is not new, the index associated with the
                final point in the old graph; we will skip this when checking for
                sameness, as they will already have been compared in the past to
                one-another, saving computation.


        Returns:
            A torch_geometric.data graph containing the observed features at
            locations, with edges between the k_n nearest neighbors.
        """
        locations_reduced, clean_ids = remove_close_points(
            np.array(locations), features, graph_delta_thresholds, old_graph_index
        )
        num_nodes = locations_reduced.shape[0]
        node_features = np.linspace(0, num_nodes - 1, num_nodes).reshape((num_nodes, 1))
        feature_mapping = {}
        feature_mapping["node_ids"] = [0, 1]

        for feature_id in features.keys():
            # Get only the features-at-points that were not removed as close/
            # redundant points
            feats = np.array([features[feature_id][i] for i in clean_ids])
            if len(feats.shape) == 1:
                feats = feats.reshape((feats.shape[0], 1))

            feature_mapping[feature_id] = [
                node_features.shape[1],
                node_features.shape[1] + feats.shape[1],
            ]
            node_features = np.column_stack((node_features, feats))

            if feature_id == "pose_vectors":
                norm = torch.tensor(feats[:, :3], dtype=torch.float)

        assert np.all(
            locations[:old_graph_index] == locations_reduced[:old_graph_index]
        ), "Old graph points shouldn't change"

        x = torch.tensor(node_features, dtype=torch.float)
        pos = torch.tensor(locations_reduced, dtype=torch.float)

        graph = Data(x=x, pos=pos, norm=norm, feature_mapping=feature_mapping)

        if k_n is not None:
            k_n = get_correct_k_n(k_n, num_nodes)

            scipy_graph = kneighbors_graph(
                locations_reduced, n_neighbors=k_n, include_self=False
            )

            scipygraph = torch_geometric.utils.from_scipy_sparse_matrix(scipy_graph)
            edge_index = scipygraph[0]

            displacements = []
            for e, edge_start in enumerate(edge_index[0]):
                edge_end = edge_index[1][e]
                displacements.append(
                    locations_reduced[edge_end] - locations_reduced[edge_start]
                )

            edge_attr = torch.tensor(np.array(displacements), dtype=torch.float)

            graph.edge_index = edge_index
            graph.edge_attr = edge_attr

        return graph

    # ------------------------ Helper --------------------------
    # ----------------------- Logging --------------------------


class GridTooSmallError(Exception):
    """Exception raised when grid is too small to fit all observations."""

    pass


class GridObjectModel(GraphObjectModel):
    """Model of an object and all its functions.

    This model has the same basic functionality as the NumpyGraph models used in older
    LM versions. On top of that we now have a grid representation of the object that
    constraints the model size and resultion. Additionally, this model class implements
    a lot of functionality that was previously implemented in the graph_utils.py file.

    TODO: General cleanups that require more changes in other code
        - remove node_ids from input_channels and have as graph attribute
        - remove .norm as attribute and store as feature instead?
    """

    def __init__(self, object_id, max_nodes, max_size, num_voxels_per_dim):
        """Initialize a grid object model.

        Args:
            object_id: id of the object
            max_nodes: maximum number of nodes in the graph. Will be k in k winner
                voxels with highest observation count.
            max_size: maximum size of the object in meters. Defines size of obejcts
                that can be represented and how locations are mapped into voxels.
            num_voxels_per_dim: number of voxels per dimension in the models grids.
                Defines the resolution of the model.
        """
        logger.info(f"init object model with id {object_id}")
        self.object_id = object_id
        self._graph = None
        self._max_nodes = max_nodes
        self._max_size = max_size  # 1=1meter
        self._num_voxels_per_dim = num_voxels_per_dim
        # Sparse, 4d torch tensors that store content in the voxels of the model grid.
        # number of observations in each voxel
        self._observation_count = None
        # Average features in each voxel with observations
        # The first 3 dims are the 3d voxel indices, the forth dimensions are features
        self._feature_grid = None
        # Average location in each voxel with observations
        # The first 3 dims are the 3d voxel indices, xyz in the fourth dimension is
        # the average location in that voxel at float precision.
        self._location_grid = None
        # For backward compatibility. May remove later on.
        # This will be true if we load a pretrained graph. If True, grids are not
        # filled or used to constrain nodes in graph.
        self.use_original_graph = False
        self._location_tree = None

    # =============== Public Interface Functions ===============
    # ------------------- Main Algorithm -----------------------
    def build_model(self, locations, features):
        """Build graph from locations and features sorted into grids."""
        (
            feature_array,
            observation_feature_mapping,
        ) = self._extract_feature_array(features)
        # TODO: part of init method?
        logger.info(f"building graph from {locations.shape[0]} observations")
        self._initialize_and_fill_grid(
            locations=locations,
            features=feature_array,
            observation_feature_mapping=observation_feature_mapping,
        )
        self._graph = self._build_graph_from_grids()
        logger.info(f"built graph {self._graph}")

    def update_model(
        self,
        locations,
        features,
        location_rel_model,
        object_location_rel_body,
        object_rotation,
    ):
        """Add new locations and features into grids and rebuild graph."""
        rf_locations, rf_features = apply_rf_transform_to_points(
            locations=locations,
            features=features,
            location_rel_model=location_rel_model,
            object_location_rel_body=object_location_rel_body,
            object_rotation=object_rotation,
        )
        (
            feature_array,
            observation_feature_mapping,
        ) = self._extract_feature_array(rf_features)
        logger.info(f"adding {locations.shape[0]} observations")
        self._update_grids(
            locations=rf_locations,
            features=feature_array,
            feature_mapping=observation_feature_mapping,
        )
        new_graph = self._build_graph_from_grids()
        assert not np.any(np.isnan(new_graph.x))
        self._graph = new_graph
        # TODO: remove eventually and do search directly in grid?
        self._location_tree = KDTree(
            new_graph.pos,
            leafsize=40,
        )

    def find_nearest_neighbors(
        self,
        search_locations,
        num_neighbors,
        return_distance=False,
    ):
        """Find nearest neighbors in graph for list of search locations.

        Note:
            This is currently using kd tree search. In the future we may consider
            doing this directly by indexing the grids. However, an initial
            implementation of this does not seem to be faster than the kd tree search
            (~5-10x slower). However one must consider that search directly in the grid
            would remove the cost of building the tree. TODO: Investigate this further.

        Returns:
            If return_distance is True, return distances. Otherwise, return indices of
            nearest neighbors.
        """
        # if self._location_tree is not None:
        # We are using the pretrained graphs and location trees for matching
        (distances, nearest_node_ids) = self._location_tree.query(
            search_locations,
            k=num_neighbors,
            p=2,  # euclidean distance
            workers=1,  # using more than 1 worker slows down run on lambda.
        )
        # else:
        #     # TODO: This is not done yet and doesn't work. It seems at the moment
        #     # That kd Tree search is still more efficient.
        #     # using new grid structure directly to query nearest neighbors
        #     distances, nearest_node_ids = self._retrieve_locations_in_radius(
        #         search_locations,
        #         search_radius=search_radius,
        #         max_neighbors=num_neighbors,
        #     )

        if return_distance:
            return distances

        return nearest_node_ids

    # ------------------ Getters & Setters ---------------------
    def set_graph(self, graph):
        """Set self._graph property and convert input graph to right format."""
        if type(graph) is not NumpyGraph:
            # could also check if is type torch_geometric.data.data.Data
            logger.debug(f"turning graph of type {type(graph)} into numpy graph")
            graph = torch_graph_to_numpy(graph)
        if self.use_original_graph:
            # Just use pretrained graph. Do not use grids to constrain nodes.
            self._graph = graph
            self._location_tree = KDTree(
                graph.pos,
                leafsize=40,
            )
        else:
            self._initialize_and_fill_grid(
                locations=graph.pos,
                features=graph.x,
                observation_feature_mapping=graph.feature_mapping,
            )
            self._graph = self._build_graph_from_grids()

    # ------------------ Logging & Saving ----------------------
    def __repr__(self) -> str:
        """Return a string representation of the object."""
        if self._graph is None:
            return f"Model for {self.object_id}:\n   No graph stored yet."
        if self._feature_grid is not None:
            grid_shape = self._feature_grid.shape
        else:
            grid_shape = 0
        repr_string = (
            f"Model for {self.object_id}:\n"
            f"   Contains {self.pos.shape[0]} points in graph.\n"
            f"   Feature grid shape: {grid_shape}\n"
            f"   Stored features and their indexes:\n"
        )
        for feature in self.feature_mapping:
            feat_ids = self.feature_mapping[feature]
            repr_string += f"           {feature} - {feat_ids[0]}:{feat_ids[1]},\n"

        return repr_string

    # ======================= Private ==========================
    # ------------------- Main Algorithm -----------------------
    def _initialize_location_mapping(self, start_location):
        """Calculate and set location_scale_factor and location_offset."""
        # scale locations to integer mappings
        voxel_size = self._max_size / self._num_voxels_per_dim
        # Find multiplier that turns voxel locations into round integer indices
        self._location_scale_factor = 1 / voxel_size
        start_index = np.array(
            np.round(start_location * self._location_scale_factor), dtype=int
        )
        # Find offset factor that places start_location at the center of the grid
        center_id = self._num_voxels_per_dim // 2
        center_voxel_index = np.array([center_id, center_id, center_id], dtype=int)
        self._location_offset = center_voxel_index - start_index

    def _initialize_and_fill_grid(
        self, locations, features, observation_feature_mapping
    ):
        # TODO: Do we still need to do this with sparse tensors?
        self._observation_count = self._generate_empty_grid(
            self._num_voxels_per_dim, n_entries=1
        )
        # initialize location mapping by calculating the scale factor and offset.
        # The offset is set such that the first observed location starts at the
        # center of the grid. To preserve the relative locations, the offset is
        # applied to all following locations.
        start_location = locations[0]
        self._initialize_location_mapping(start_location=start_location)
        # initialize self._feature_grid with feat_dim calculated from features
        feat_dim = features.shape[-1]
        self._feature_grid = self._generate_empty_grid(
            self._num_voxels_per_dim, n_entries=feat_dim
        )
        self._location_grid = self._generate_empty_grid(
            self._num_voxels_per_dim, n_entries=3
        )
        # increment counters in observation_count
        self._update_grids(
            locations=locations,
            features=features,
            feature_mapping=observation_feature_mapping,
        )

    def _update_grids(self, locations, features, feature_mapping):
        """Update count, location and feature grids with observations.

        Raises:
            GridTooSmallError: If too many observations are outside of the grid
        """
        location_grid_ids = self._locations_to_grid_ids(locations)
        locations_in_bounds = np.all(
            (location_grid_ids >= 0) & (location_grid_ids < self._num_voxels_per_dim),
            axis=1,
        )
        percent_in_bounds = sum(locations_in_bounds) / len(locations_in_bounds)
        if percent_in_bounds < 0.9:
            logger.info(
                "Too many observations outside of grid "
                f"({np.round(percent_in_bounds * 100, 2)}%). Skipping update of grids."
            )
            raise GridTooSmallError
        voxel_ids_of_new_obs = location_grid_ids[locations_in_bounds]
        self._observation_count = increment_sparse_tensor_by_count(
            self._observation_count, voxel_ids_of_new_obs
        )

        # if new features contain input channel or features, add them to mapping
        if self.feature_mapping is not None:
            updated_fm, new_feat_dim = self._update_feature_mapping(feature_mapping)
        else:  # no graph has been initialized yet
            updated_fm = feature_mapping
            new_feat_dim = features.shape[-1]

        # Since we have dense entries in the last dimension that we want associated
        # with the voxel id in the first 3 dims, we need to extract them a bit
        # tediously to do the averaging. TODO: Maybe there is a better way to do this
        new_features = []
        previous_features_at_indices = []
        new_locations = []
        previous_locations_at_indices = []
        new_indices = []

        locations = locations[locations_in_bounds]
        features = features[locations_in_bounds]
        # update average features for each voxel with content
        # The indices() function will give us the voxel indices that have values
        voxels_with_values = self._observation_count.indices()
        # Calculate average of new features and put in new_feature_grid
        for voxel in zip(
            voxels_with_values[0],
            voxels_with_values[1],
            voxels_with_values[2],
        ):
            observations_in_voxel_ids = np.where(
                (voxel_ids_of_new_obs == voxel).all(axis=1)
            )[0]
            # Only update if there are new observations for this voxel
            if len(observations_in_voxel_ids) > 0:
                new_indices.append(np.array(voxel))
                previous_loc_in_voxel = get_values_from_dense_last_dim(
                    self._location_grid, voxel
                )
                previous_locations_at_indices.append(previous_loc_in_voxel)
                locations_in_voxel = locations[observations_in_voxel_ids]
                new_avg_location = self._get_new_voxel_location(
                    locations_in_voxel,
                    previous_loc_in_voxel,
                    voxel,
                )
                new_locations.append(new_avg_location)

                previous_feat_in_voxel = get_values_from_dense_last_dim(
                    self._feature_grid, voxel
                )

                previous_features_at_indices.append(previous_feat_in_voxel)
                features_in_voxel = features[observations_in_voxel_ids]
                new_avg_feat = self._get_new_voxel_features(
                    features_in_voxel,
                    np.array(previous_feat_in_voxel),
                    voxel,
                    feature_mapping,
                    updated_fm,
                    new_feat_dim,
                )
                new_features.append(new_avg_feat)

        (
            prev_sparse_locs,
            new_sparse_locs,
        ) = self._old_new_lists_to_sparse_tensors(
            indices=new_indices,
            new_values=new_locations,
            old_values=previous_locations_at_indices,
            target_mat_shape=self._location_grid.shape,
        )
        # Subtract old locations since new ones already contain them in their average
        # Don't just overwrite with new_sparse_locs since we may have voxels in the
        # _location_grid that did not get updated and should not be set to 0 now.
        self._location_grid = self._location_grid - prev_sparse_locs + new_sparse_locs

        (
            prev_sparse_feats,
            new_sparse_feats,
        ) = self._old_new_lists_to_sparse_tensors(
            new_indices,
            new_features,
            previous_features_at_indices,
            self._feature_grid.shape,
        )
        self._feature_grid = self._feature_grid - prev_sparse_feats + new_sparse_feats
        self._current_feature_mapping = updated_fm

    def _build_graph_from_grids(self):
        """Build graph from grids by taking the top k voxels with content.

        Returns:
            Graph with locations and features at the top k voxels with content.
        """
        top_voxel_idxs = self._get_top_k_voxel_indices()

        locations_at_ids = self._location_grid.to_dense()[
            top_voxel_idxs[0], top_voxel_idxs[1], top_voxel_idxs[2]
        ]
        features_at_ids = self._feature_grid.to_dense()[
            top_voxel_idxs[0], top_voxel_idxs[1], top_voxel_idxs[2]
        ]
        graph = build_point_cloud_graph(
            locations=np.array(locations_at_ids),
            features=np.array(features_at_ids),
            feature_mapping=self._current_feature_mapping,
        )
        # TODO: remove eventually and do search directly in grid?
        self._location_tree = KDTree(
            graph.pos,
            leafsize=40,
        )
        return graph

    # ------------------------ Helper --------------------------
    def _extract_feature_array(self, feature_dict):
        """Turns the dict of features into an array + feature mapping.

        For efficient calculations all features are stored in a single array. To
        retain information about where a specific feature is stored we have the
        feature_mapping dictionary which stores the indices of where each feature is
        stored. This function extracts all features in feature_dict and stacks them in
        a matrix. The returned feature_mapping tells where in the array each feature
        is stored.

        Note:
            Once this class goes out of the experimental stage and becomes more the
            default, the buffer get_all_features_on_object function could directly
            return this format instead of having to convert twice.

        Returns:
            feature_array: Array of features.
            feature_mapping: Dictionary with feature names as keys and their
                corresponding indices in feature_array as values.
        """
        feature_mapping = {}
        feature_array = None
        for feature in feature_dict.keys():
            feats = feature_dict[feature]

            if len(feats.shape) == 1:
                feats = feats.reshape((feats.shape[0], 1))

            if feature_array is None:
                feature_array = feats
                prev_feat_shape = 0
            else:
                prev_feat_shape = feature_array.shape[1]
                feature_array = np.column_stack((feature_array, feats))

            feature_mapping[feature] = [
                prev_feat_shape,
                prev_feat_shape + feats.shape[1],
            ]
        return feature_array, feature_mapping

    def _generate_empty_grid(self, num_voxels, n_entries):
        # NOTE: torch sparse is made for 2D tensors. We use it for 4D tensors.
        # Some operations may not work as expected on these.
        shape = (num_voxels, num_voxels, num_voxels, n_entries)
        # Create empty sparse tensor
        sparse_tensor = torch.sparse_coo_tensor(
            torch.zeros((4, 0), dtype=torch.long), torch.tensor([]), size=shape
        )
        return sparse_tensor.coalesce()

    def _get_new_voxel_location(
        self, new_locations_in_voxel, previous_loc_in_voxel, voxel
    ):
        """Calculate new average location for a voxel.

        Returns:
            New average location for a voxel.
        """
        avg_loc = np.mean(new_locations_in_voxel, axis=0)
        # Only average with previous location if there was one stored there before.
        # since self._observation_count already includes the new observations it needs
        # to be > the number of new observations in the voxel.
        if self._observation_count[voxel[0], voxel[1], voxel[2], 0] > len(
            new_locations_in_voxel
        ):
            # NOTE: could weight these
            avg_loc = (avg_loc + previous_loc_in_voxel) / 2
        return avg_loc

    def _get_new_voxel_features(
        self,
        new_features_in_voxel,
        previous_feat_in_voxel,
        voxel,
        obs_fm,
        target_fm,
        target_feat_dim,
    ):
        """Calculate new average features for a voxel.

        Returns:
            New average features for a voxel.
        """
        new_feature_avg = np.zeros(target_feat_dim)
        if ("pose_vectors" in obs_fm.keys()) and (
            "pose_fully_defined" in obs_fm.keys()
        ):
            # TODO: deal with case where not all of those keys are present
            pv_ids = obs_fm["pose_vectors"]
            pdefined_ids = obs_fm["pose_fully_defined"]
            pose_vecs = new_features_in_voxel[:, pv_ids[0] : pv_ids[1]]
            pdefined = new_features_in_voxel[:, pdefined_ids[0] : pdefined_ids[1]]
            pv_mean, use_cds_to_update = pose_vector_mean(pose_vecs, pdefined)
        for feature in obs_fm:
            ids = obs_fm[feature]
            feats = new_features_in_voxel[:, ids[0] : ids[1]]
            if feature == "hsv":
                avg_feat = np.zeros(3)
                avg_feat[0] = circular_mean(feats[:, 0])
                avg_feat[1:] = np.mean(feats[:, 1:], axis=0)
            elif feature == "pose_vectors":
                avg_feat = pv_mean
            elif feature in ["on_object", "pose_fully_defined"]:
                avg_feat = get_most_common_bool(feats)
                # NOTE: object_id may need its own most common function until
                # IDs actually represent similarities
            elif feature == "object_id":
                avg_feat = get_most_common_value(feats)
            else:
                avg_feat = np.mean(feats, axis=0)
            # Only take average if there was a feature stored here before.
            # since self._observation_count already includes the new obs
            # this needs to be > the number of new feature obs in the voxel.
            num_obs_in_voxel = self._observation_count[voxel[0], voxel[1], voxel[2], 0]
            num_new_obs = len(feats)
            if num_obs_in_voxel > num_new_obs:
                old_ids = self.feature_mapping[feature]
                previous_average = previous_feat_in_voxel[old_ids[0] : old_ids[1],]
                num_old_obs = num_obs_in_voxel - num_new_obs

                if feature == "pose_vectors":
                    if avg_feat is None:
                        avg_feat = previous_average
                    elif use_cds_to_update is False:
                        avg_feat[3:] = previous_average[3:]
                elif feature == "object_id":
                    # TODO: Figure out a more nuanced way to take into account past obs
                    if avg_feat != previous_average:
                        if num_old_obs > num_new_obs:
                            avg_feat = previous_average
                        else:
                            previous_average = avg_feat
                # NOTE: could weight these
                avg_feat = (avg_feat + previous_average) / 2
            target_ids = target_fm[feature]
            new_feature_avg[target_ids[0] : target_ids[1]] = avg_feat
        return new_feature_avg

    def _update_feature_mapping(self, new_fm):
        """Update feature_mapping dict with potentially new features.

        Returns:
            updated_fm: Updated feature_mapping dictionary.
            new_feature_dim: Dimension of the new features.
        """
        updated_fm = copy.deepcopy(self.feature_mapping)
        new_feature_dim = self.x.shape[-1]
        for feature in new_fm:
            if feature not in updated_fm:
                start_idx = new_feature_dim
                new_feature_idxs = new_fm[feature]
                feat_size = new_feature_idxs[1] - new_feature_idxs[0]
                stop_idx = new_feature_dim + feat_size
                updated_fm[feature] = [start_idx, stop_idx]
                new_feature_dim = stop_idx
        return updated_fm, new_feature_dim

    def _get_top_k_voxel_indices(self):
        """Get voxel indices with k highest observation counts.

        Note:
            May return less than k (self._max_nodes) voxels if there are
            less than k voxels with content.

        Returns:
            Indices of the top k voxels with content.
        """
        num_non_zero_voxels = len(self._observation_count.values())
        if num_non_zero_voxels < self._max_nodes:
            print("There are less than max_nodes voxels with content.")
            k = num_non_zero_voxels
        else:
            k = self._max_nodes
        _counts, top_k_indices = self._observation_count.values().topk(k)
        return self._observation_count.indices()[:3, top_k_indices]

    def _locations_to_grid_ids(self, locations):
        """Convert locations to grid ids using scale_factor and location_offset.

        Returns:
            Grid ids for the locations.
        """
        return np.array(
            np.round(locations * self._location_scale_factor) + self._location_offset,
            dtype=int,
        )

    def _old_new_lists_to_sparse_tensors(
        self, indices, new_values, old_values, target_mat_shape
    ):
        """Turn two lists of old and new values into sparse tensors.

        Args:
            indices: list of indices of the form [x, y, z] (Will be expanded
                to 4d for sparse tensor)
            new_values: list of new values to be put into sparse tensor
            old_values: list of old values to be put into another sparse tensor
            target_mat_shape: shape of the two sparse tensors

        Returns:
            old_sparse_mat: Sparse tensor with old values.
            new_sparse_mat: Sparse tensor with new values.
        """
        indices_4d = expand_index_dims(indices, last_dim_size=target_mat_shape[-1])
        new_sparse_mat = torch.sparse_coo_tensor(
            indices_4d,
            np.array(new_values).flatten(),
            target_mat_shape,
        ).coalesce()
        old_sparse_mat = torch.sparse_coo_tensor(
            indices_4d,
            np.array(old_values).flatten(),
            target_mat_shape,
        ).coalesce()
        return old_sparse_mat, new_sparse_mat

    # ----------------------- Logging --------------------------
