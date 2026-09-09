# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)

MAX_DISTANCE = 0.01


def graph_with_nearest(distance: float) -> MagicMock:
    # A stored object model whose nearest point is `distance` away from any query.
    graph = MagicMock()
    graph.find_nearest_neighbors.return_value = np.array([distance])
    return graph


def lm_with_model(channel_distances: dict[str, float]) -> MagicMock:
    # An EvidenceGraphLM stand-in knowing one object, "mug", with a stored
    # graph per input channel; the real methods under test are bound to it.
    lm = MagicMock(spec=EvidenceGraphLM)
    lm.graph_memory = MagicMock()
    lm.output_max_model_distance = MAX_DISTANCE
    lm.get_all_known_object_ids.return_value = ["mug"]
    lm.get_input_channels_in_graph.return_value = list(channel_distances)
    lm.graph_memory.get_graph.side_effect = lambda _graph_id, channel: (
        graph_with_nearest(channel_distances[channel])
    )
    return lm


def mlh(graph_id: str = "mug", location=(0.0, 0.0, 0.0)) -> dict:
    return {
        "graph_id": graph_id,
        "location": None if location is None else np.asarray(location),
        "rotation": Rotation.identity(),
        "evidence": 3.0,
    }


class MlhOnModelTest(unittest.TestCase):
    def test_true_within_the_distance_of_a_stored_point(self) -> None:
        lm = lm_with_model({"patch_0": MAX_DISTANCE / 2})
        self.assertTrue(EvidenceGraphLM._mlh_on_model(lm, mlh()))

    def test_true_exactly_at_the_distance(self) -> None:
        lm = lm_with_model({"patch_0": MAX_DISTANCE})
        self.assertTrue(EvidenceGraphLM._mlh_on_model(lm, mlh()))

    def test_false_beyond_the_distance(self) -> None:
        lm = lm_with_model({"patch_0": MAX_DISTANCE * 1.5})
        self.assertFalse(EvidenceGraphLM._mlh_on_model(lm, mlh()))

    def test_the_nearest_channel_decides(self) -> None:
        lm = lm_with_model({"patch_0": MAX_DISTANCE * 5, "learning_module_0": 0.0})
        self.assertTrue(EvidenceGraphLM._mlh_on_model(lm, mlh()))

    def test_false_for_an_unknown_object(self) -> None:
        lm = lm_with_model({"patch_0": 0.0})
        self.assertFalse(EvidenceGraphLM._mlh_on_model(lm, mlh(graph_id="new_object0")))

    def test_false_without_a_location(self) -> None:
        lm = lm_with_model({"patch_0": 0.0})
        self.assertFalse(EvidenceGraphLM._mlh_on_model(lm, mlh(location=None)))

    def test_queries_the_stored_graph_with_the_mlh_location(self) -> None:
        graph = graph_with_nearest(0.0)
        lm = lm_with_model({"patch_0": 0.0})
        lm.graph_memory.get_graph.side_effect = None
        lm.graph_memory.get_graph.return_value = graph

        EvidenceGraphLM._mlh_on_model(lm, mlh(location=(1.0, 2.0, 3.0)))

        lm.graph_memory.get_graph.assert_called_once_with("mug", "patch_0")
        (query,), kwargs = graph.find_nearest_neighbors.call_args
        np.testing.assert_array_equal(query, [[1.0, 2.0, 3.0]])
        self.assertEqual(kwargs, {"num_neighbors": 1, "return_distance": True})


def lm_for_output(
    *,
    on_object: bool,
    terminal_state: str | None,
    on_model: bool,
    min_evidence: float = 0.0,
):
    # Everything get_output reads, with the gate inputs controllable; the MLH
    # carries evidence 3.0.
    lm = MagicMock(spec=EvidenceGraphLM)
    lm.learning_module_id = "learning_module_1"
    lm.output_min_evidence = min_evidence
    lm.terminal_state = terminal_state
    lm.buffer = MagicMock()
    lm.buffer.get_currently_on_object.return_value = on_object
    lm.buffer.__len__.return_value = 4
    lm._get_current_mlh.return_value = mlh()
    lm._object_pose_to_features.return_value = np.eye(3)
    lm._object_id_to_features.return_value = 321
    lm._enough_symmetry_evidence_accumulated.return_value = False
    lm._mlh_on_model.return_value = on_model
    return lm


class GetOutputGateTest(unittest.TestCase):
    def test_passes_when_matched_on_object_and_on_the_model(self) -> None:
        message = EvidenceGraphLM.get_output(
            lm_for_output(on_object=True, terminal_state="match", on_model=True)
        )
        self.assertTrue(message.pass_message)
        self.assertEqual(message.non_morphological_features["object_id"], 321)

    def test_withheld_off_the_model(self) -> None:
        message = EvidenceGraphLM.get_output(
            lm_for_output(on_object=True, terminal_state="match", on_model=False)
        )
        self.assertFalse(message.pass_message)

    def test_withheld_without_a_match(self) -> None:
        message = EvidenceGraphLM.get_output(
            lm_for_output(on_object=True, terminal_state=None, on_model=True)
        )
        self.assertFalse(message.pass_message)

    def test_withheld_off_the_object(self) -> None:
        message = EvidenceGraphLM.get_output(
            lm_for_output(on_object=False, terminal_state="match", on_model=True)
        )
        self.assertFalse(message.pass_message)

    def test_passes_at_the_evidence_floor(self) -> None:
        lm = lm_for_output(
            on_object=True, terminal_state="match", on_model=True, min_evidence=3.0
        )
        self.assertTrue(EvidenceGraphLM.get_output(lm).pass_message)

    def test_withheld_below_the_evidence_floor(self) -> None:
        lm = lm_for_output(
            on_object=True, terminal_state="match", on_model=True, min_evidence=3.5
        )
        self.assertFalse(EvidenceGraphLM.get_output(lm).pass_message)
        lm._mlh_on_model.assert_not_called()

    def test_the_model_check_is_skipped_unless_the_other_gates_pass(self) -> None:
        lm = lm_for_output(on_object=True, terminal_state=None, on_model=True)
        EvidenceGraphLM.get_output(lm)
        lm._mlh_on_model.assert_not_called()
