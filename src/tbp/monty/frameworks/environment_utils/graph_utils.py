# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import numpy as np


def get_edge_index(graph, previous_node, new_node):
    """Get the edge index between two nodes in a graph.

    TODO: There must be an easier way to do this!

    Args:
        graph: torch_geometric.data graph
        previous_node: node ID if the first node in the graph
        new_node: node ID if the second node in the graph

    Returns:
        edge ID between the two nodes
    """
    edges_of_node = np.where(graph.edge_index[0] == previous_node)[0]
    for i in range(len(edges_of_node)):
        possible_next_node = graph.edge_index[1][edges_of_node[i]]
        if possible_next_node == new_node:
            return edges_of_node[i]
