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


def get_edge_index(graph, previous_node, new_node) -> int | None:
    """Return the edge index between two nodes in a graph.

    Args:
        graph: torch_geometric.data graph.
        previous_node: Node ID of the first node in the graph.
        new_node: Node ID of the second node in the graph.

    Returns:
        Edge ID between the two nodes, or None if no such edge exists.
    """
    mask = (graph.edge_index[0] == previous_node) & (graph.edge_index[1] == new_node)
    if mask.any():
        return mask.nonzero().view(-1)[0].item()
    return None
