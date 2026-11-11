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
from dataclasses import dataclass
from unittest.mock import MagicMock, patch, sentinel

import numpy as np
import numpy.testing as nptest
from hypothesis import given
from hypothesis import strategies as st
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)

MAX_MATCH_DISTANCE = 0.01
TERMINAL_STATES = [None, "match", "no_match", "time_out", "pose_time_out"]

terminal_states = st.sampled_from(TERMINAL_STATES)

not_match_terminal_states = st.sampled_from(
    [state for state in TERMINAL_STATES if state != "match"]
)


@dataclass
class GatingConditions:
    on_object: bool
    terminal_state: str | None
    mlh_on_model: bool


off_object_gating_conditions = st.builds(
    GatingConditions,
    on_object=st.just(value=False),
    terminal_state=terminal_states,
    mlh_on_model=st.booleans(),
)

not_match_gating_conditions = st.builds(
    GatingConditions,
    on_object=st.just(value=True),
    terminal_state=not_match_terminal_states,
    mlh_on_model=st.booleans(),
)

off_model_gating_conditions = st.builds(
    GatingConditions,
    on_object=st.just(value=True),
    terminal_state=st.just("match"),
    mlh_on_model=st.just(value=False),
)


class GetOutputTest(unittest.TestCase):
    def setUp(self) -> None:
        # get_output reads the rotation, graph_id (hashed, so it must be a str)
        # and evidence; the location only matters to _mlh_on_model, which is
        # patched.
        self.mlh = {
            "graph_id": "mug",
            "location": np.zeros(3),
            "rotation": Rotation.identity(),
            "evidence": 1.0,
        }

    @patch(
        "tbp.monty.frameworks.models.evidence_matching.learning_module.EvidenceGraphLM._mlh_on_model"
    )
    @patch(
        "tbp.monty.frameworks.models.evidence_matching.learning_module.EvidenceGraphLM._get_current_mlh"
    )
    def test_calls_mlh_on_model_with_current_mlh(
        self,
        get_current_mlh_mock: MagicMock,
        mlh_on_model_mock: MagicMock,
    ) -> None:
        get_current_mlh_mock.return_value = self.mlh
        lm = EvidenceGraphLM(
            max_match_distance=MAX_MATCH_DISTANCE, tolerances={}, feature_weights={}
        )
        lm.buffer = MagicMock()
        lm.buffer.get_currently_on_object.return_value = True
        lm.terminal_state = "match"

        lm.get_output()

        mlh_on_model_mock.assert_called_once_with(self.mlh)

    @patch(
        "tbp.monty.frameworks.models.evidence_matching.learning_module.EvidenceGraphLM._mlh_on_model",
        return_value=True,
    )
    @patch(
        "tbp.monty.frameworks.models.evidence_matching.learning_module.EvidenceGraphLM._get_current_mlh"
    )
    def test_returns_message_that_passes_when_on_object_matched_and_mlh_on_model(
        self,
        get_current_mlh_mock: MagicMock,
        mlh_on_model_mock: MagicMock,  # noqa: ARG002
    ) -> None:
        get_current_mlh_mock.return_value = self.mlh
        lm = EvidenceGraphLM(
            max_match_distance=MAX_MATCH_DISTANCE, tolerances={}, feature_weights={}
        )
        lm.buffer = MagicMock()
        lm.buffer.get_currently_on_object.return_value = True
        lm.terminal_state = "match"

        message = lm.get_output()

        assert message is not None, "get_output should return a message"
        self.assertTrue(message.pass_message)

    @given(
        gating_conditions=st.one_of(
            off_object_gating_conditions,
            not_match_gating_conditions,
            off_model_gating_conditions,
        )
    )
    @patch(
        "tbp.monty.frameworks.models.evidence_matching.learning_module.EvidenceGraphLM._mlh_on_model"
    )
    @patch(
        "tbp.monty.frameworks.models.evidence_matching.learning_module.EvidenceGraphLM._get_current_mlh"
    )
    def test_returns_message_that_does_not_pass_unless_on_object_matched_and_mlh_on_model(  # noqa: E501
        self,
        get_current_mlh_mock: MagicMock,
        mlh_on_model_mock: MagicMock,
        gating_conditions: GatingConditions,
    ) -> None:
        get_current_mlh_mock.return_value = self.mlh
        mlh_on_model_mock.return_value = gating_conditions.mlh_on_model
        lm = EvidenceGraphLM(
            max_match_distance=MAX_MATCH_DISTANCE, tolerances={}, feature_weights={}
        )
        lm.buffer = MagicMock()
        lm.buffer.get_currently_on_object.return_value = gating_conditions.on_object
        lm.terminal_state = gating_conditions.terminal_state

        message = lm.get_output()

        assert message is not None, "get_output should return a message"
        self.assertFalse(message.pass_message)

    @given(
        gating_conditions=st.one_of(
            off_object_gating_conditions, not_match_gating_conditions
        )
    )
    @patch(
        "tbp.monty.frameworks.models.evidence_matching.learning_module.EvidenceGraphLM._mlh_on_model"
    )
    @patch(
        "tbp.monty.frameworks.models.evidence_matching.learning_module.EvidenceGraphLM._get_current_mlh"
    )
    def test_does_not_call_mlh_on_model_when_off_object_or_unmatched(
        self,
        get_current_mlh_mock: MagicMock,
        mlh_on_model_mock: MagicMock,
        gating_conditions: GatingConditions,
    ) -> None:
        get_current_mlh_mock.return_value = self.mlh
        lm = EvidenceGraphLM(
            max_match_distance=MAX_MATCH_DISTANCE, tolerances={}, feature_weights={}
        )
        lm.buffer = MagicMock()
        lm.buffer.get_currently_on_object.return_value = gating_conditions.on_object
        lm.terminal_state = gating_conditions.terminal_state

        lm.get_output()

        mlh_on_model_mock.assert_not_called()


MAX_INPUT_CHANNELS = 4
MAX_BEYOND_DISTANCE = 100 * MAX_MATCH_DISTANCE

distances_within_max_match = st.floats(min_value=0.0, max_value=MAX_MATCH_DISTANCE)

distances_beyond_max_match = st.floats(
    min_value=MAX_MATCH_DISTANCE, max_value=MAX_BEYOND_DISTANCE, exclude_min=True
)


@dataclass
class ChannelDistances:
    distances: list[float]
    mlh_on_model: bool


@st.composite
def channel_distances(draw: st.DrawFn) -> ChannelDistances:
    mlh_on_model = draw(st.booleans())
    num_channels = draw(st.integers(min_value=1, max_value=MAX_INPUT_CHANNELS))

    if mlh_on_model:
        distances = draw(
            st.lists(
                st.one_of(distances_within_max_match, distances_beyond_max_match),
                min_size=num_channels,
                max_size=num_channels,
            )
        )
        near_channel = draw(st.integers(min_value=0, max_value=num_channels - 1))
        distances[near_channel] = draw(distances_within_max_match)
    else:
        distances = draw(
            st.lists(
                distances_beyond_max_match, min_size=num_channels, max_size=num_channels
            )
        )

    return ChannelDistances(distances=distances, mlh_on_model=mlh_on_model)


class MlhOnModelTest(unittest.TestCase):
    @patch(
        "tbp.monty.frameworks.models.evidence_matching.learning_module.EvidenceGraphLM.get_all_known_object_ids",
        return_value=[sentinel.graph_id],
    )
    def test_returns_false_when_mlh_has_no_location(
        self,
        get_all_known_object_ids_mock: MagicMock,  # noqa: ARG002
    ) -> None:
        lm = EvidenceGraphLM(
            max_match_distance=MAX_MATCH_DISTANCE, tolerances={}, feature_weights={}
        )
        lm.graph_memory = MagicMock()

        on_model = lm._mlh_on_model({"graph_id": sentinel.graph_id, "location": None})

        self.assertFalse(on_model)

    @patch(
        "tbp.monty.frameworks.models.evidence_matching.learning_module.EvidenceGraphLM.get_all_known_object_ids",
        return_value=[sentinel.graph_id],
    )
    def test_returns_false_when_mlh_object_is_unknown(
        self,
        get_all_known_object_ids_mock: MagicMock,  # noqa: ARG002
    ) -> None:
        lm = EvidenceGraphLM(
            max_match_distance=MAX_MATCH_DISTANCE, tolerances={}, feature_weights={}
        )
        lm.graph_memory = MagicMock()

        on_model = lm._mlh_on_model(
            {"graph_id": sentinel.unknown_graph_id, "location": np.zeros(3)}
        )

        self.assertFalse(on_model)

    @patch(
        "tbp.monty.frameworks.models.evidence_matching.learning_module.EvidenceGraphLM.get_input_channels_in_graph",
        return_value=[sentinel.channel_0],
    )
    @patch(
        "tbp.monty.frameworks.models.evidence_matching.learning_module.EvidenceGraphLM.get_all_known_object_ids",
        return_value=[sentinel.graph_id],
    )
    def test_calls_find_nearest_neighbors_with_mlh_location(
        self,
        get_all_known_object_ids_mock: MagicMock,  # noqa: ARG002
        get_input_channels_in_graph_mock: MagicMock,  # noqa: ARG002
    ) -> None:
        lm = EvidenceGraphLM(
            max_match_distance=MAX_MATCH_DISTANCE, tolerances={}, feature_weights={}
        )
        graph_memory_mock = MagicMock()
        graph_mock = graph_memory_mock.get_graph.return_value
        graph_mock.find_nearest_neighbors.return_value = np.array([0.0])
        lm.graph_memory = graph_memory_mock
        mlh_location_sentinel = np.array([1.0, 2.0, 3.0])

        lm._mlh_on_model(
            {"graph_id": sentinel.graph_id, "location": mlh_location_sentinel}
        )

        graph_memory_mock.get_graph.assert_called_once_with(
            sentinel.graph_id, sentinel.channel_0
        )
        (query,), kwargs = graph_mock.find_nearest_neighbors.call_args
        nptest.assert_array_equal(query, [mlh_location_sentinel])
        self.assertEqual(kwargs, {"num_neighbors": 1, "return_distance": True})

    @given(channel_distances=channel_distances())
    @patch(
        "tbp.monty.frameworks.models.evidence_matching.learning_module.EvidenceGraphLM.get_input_channels_in_graph"
    )
    @patch(
        "tbp.monty.frameworks.models.evidence_matching.learning_module.EvidenceGraphLM.get_all_known_object_ids",
        return_value=[sentinel.graph_id],
    )
    def test_returns_whether_any_input_channel_is_within_max_match_distance(
        self,
        get_all_known_object_ids_mock: MagicMock,  # noqa: ARG002
        get_input_channels_in_graph_mock: MagicMock,
        channel_distances: ChannelDistances,
    ) -> None:
        """`channel_distances.mlh_on_model` is correct by construction."""
        get_input_channels_in_graph_mock.return_value = [
            getattr(sentinel, f"channel_{i}")
            for i in range(len(channel_distances.distances))
        ]
        graph_mocks = []
        for distance in channel_distances.distances:
            graph_mock = MagicMock()
            graph_mock.find_nearest_neighbors.return_value = np.array([distance])
            graph_mocks.append(graph_mock)
        lm = EvidenceGraphLM(
            max_match_distance=MAX_MATCH_DISTANCE, tolerances={}, feature_weights={}
        )
        lm.graph_memory = MagicMock()
        lm.graph_memory.get_graph.side_effect = graph_mocks

        on_model = lm._mlh_on_model(
            {"graph_id": sentinel.graph_id, "location": np.zeros(3)}
        )

        self.assertEqual(on_model, channel_distances.mlh_on_model)
