# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


from typing import cast

import torch

from tbp.monty.frameworks.models.object_model import GraphObjectModel


def assert_graph_object_models_equal(
    left: GraphObjectModel, right: GraphObjectModel
) -> None:
    """Custom assertion comparison for GraphObjectModel.

    Args:
        left: The left GraphObjectModel instance to compare with.
        right: The right GraphObjectModel instance to compare with.

    Raises:
        AssertionError: If the two GraphObjectModel instances are not equal.
    """
    if left._graph is None and right._graph is None:
        raise AssertionError("Both GraphObjectModel instances have no graph.")

    if left._graph is None or right._graph is None:
        raise AssertionError("One of the GraphObjectModel instances has no graph.")

    if set(left._graph.keys) != set(right._graph.keys):
        raise AssertionError(
            "The keys of the two GraphObjectModel instances are not equal.\n"
            f"Left keys: {set(left._graph.keys)}\n"
            f"Right keys: {set(right._graph.keys)}\n"
        )

    for key in left._graph.keys:
        v_left, v_right = left._graph[key], right._graph[key]

        if torch.is_tensor(v_left):
            if not torch.equal(v_left, v_right):
                raise AssertionError(
                    f"The {key} values of the two GraphObjectModel instances are not "
                    f"equal.\nLeft value: {v_left}\n"
                    f"Right value: {v_right}\n"
                )
        elif v_left != v_right:
            raise AssertionError(
                f"The {key} values of the two GraphObjectModel instances are not "
                f"equal.\nLeft value: {v_left}\n"
                f"Right value: {v_right}"
            )


def assert_trained_models_equal(serial_model: dict, parallel_model: dict) -> None:
    """Custom assertion comparison for trained models.

    Args:
        serial_model: The serial model to compare with.
        parallel_model: The parallel model to compare with.

    Raises:
        AssertionError: If the two trained models are not equal.
    """
    if set(parallel_model["lm_dict"].keys()) != set(serial_model["lm_dict"].keys()):
        raise AssertionError("LM IDs do not match")

    for lm_id in parallel_model["lm_dict"].keys():
        p = parallel_model["lm_dict"][lm_id]
        s = serial_model["lm_dict"][lm_id]
        if set(p.keys()) != set(s.keys()):
            raise AssertionError(f"LM {lm_id} keys do not match")

        p_graph_memory = p["graph_memory"]
        s_graph_memory = s["graph_memory"]
        if set(p_graph_memory.keys()) != set(s_graph_memory.keys()):
            raise AssertionError(f"LM {lm_id} graph memory keys do not match")

        for graph_id in p_graph_memory.keys():
            p_graph = p_graph_memory[graph_id]
            s_graph = s_graph_memory[graph_id]
            if set(p_graph.keys()) != set(s_graph.keys()):
                raise AssertionError(f"LM {lm_id} graph {graph_id} keys do not match")

            for channel_id in p_graph.keys():
                p_graph_data: GraphObjectModel = cast(
                    "GraphObjectModel", p_graph[channel_id]
                )
                s_graph_data: GraphObjectModel = cast(
                    "GraphObjectModel", s_graph[channel_id]
                )
                assert_graph_object_models_equal(p_graph_data, s_graph_data)
