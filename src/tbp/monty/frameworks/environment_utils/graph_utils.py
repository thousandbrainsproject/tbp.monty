# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


def get_edge_index(graph, previous_node, new_node):
    """Get the edge index between two nodes in a graph.

    Args:
        graph: torch_geometric.data graph
        previous_node: node ID if the first node in the graph
        new_node: node ID if the second node in the graph

    Returns:
        edge ID between the two nodes
    """
    mask = (graph.edge_index[0] == previous_node) & (graph.edge_index[1] == new_node)
    edge_ids = mask.nonzero().view(-1)
    return edge_ids[0].item() if len(edge_ids) > 0 else None
