# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np

from tbp.monty.frameworks.models.evidence_matching.hypotheses import Hypotheses
from tbp.monty.frameworks.models.evidence_matching.hypotheses_displacer import (
    DefaultHypothesesDisplacer,
)


class DefaultHypothesesDisplacerTest(TestCase):
    """Tests for the DefaultHypothesesDisplacer with unified hypothesis space."""

    def setUp(self) -> None:
        self.mock_graph_memory = Mock()
        self.mock_graph_memory.get_input_channels_in_graph = Mock(
            return_value=["channel_a", "channel_b"]
        )

        self.displacer = DefaultHypothesesDisplacer(
            feature_weights={
                "channel_a": {"pose_vectors": [1, 1]},
                "channel_b": {"pose_vectors": [1, 1]},
            },
            graph_memory=self.mock_graph_memory,
            max_match_distance=0.01,
            tolerances={
                "channel_a": {},
                "channel_b": {},
            },
            use_features_for_matching={"channel_a": False, "channel_b": False},
            past_weight=1,
            present_weight=1,
        )

    def test_multi_channel_evidence_sums(self) -> None:
        """Test that evidence from two channels is summed into unified hypotheses.

        Sets up two channels each returning known evidence arrays, and verifies
        the total evidence is the sum of per-channel contributions.
        """
        num_hyps = 3
        hypotheses = Hypotheses(
            evidence=np.array([1.0, 2.0, 3.0]),
            locations=np.zeros((num_hyps, 3)),
            poses=np.tile(np.eye(3), (num_hyps, 1, 1)),
            possible=np.ones(num_hyps, dtype=bool),
        )

        # Channel A returns evidence [0.5, 0.5, 0.5]
        # Channel B returns evidence [1.0, 0.0, -1.0]
        evidence_by_channel = {
            "channel_a": np.array([0.5, 0.5, 0.5]),
            "channel_b": np.array([1.0, 0.0, -1.0]),
        }
        with patch.object(
            self.displacer,
            "_calculate_evidence_for_new_locations",
            side_effect=lambda **kw: evidence_by_channel[kw["input_channel"]],
        ):
            result, _telemetry = (
                self.displacer.displace_hypotheses_and_compute_evidence(
                    displacements={
                        "channel_a": np.zeros(3),
                        "channel_b": np.zeros(3),
                    },
                    features={
                        "channel_a": {"pose_fully_defined": True},
                        "channel_b": {"pose_fully_defined": True},
                    },
                    evidence_update_threshold=-np.inf,
                    graph_id="test_object",
                    possible_hypotheses=hypotheses,
                    total_hypotheses_count=num_hyps,
                )
            )

        # Expected: past_weight * old_evidence + present_weight * summed_new
        # summed_new = [0.5+1.0, 0.5+0.0, 0.5+(-1.0)] = [1.5, 0.5, -0.5]
        # result = 1 * [1.0, 2.0, 3.0] + 1 * [1.5, 0.5, -0.5] = [2.5, 2.5, 2.5]
        np.testing.assert_array_almost_equal(result.evidence, [2.5, 2.5, 2.5])

    def test_prediction_error_computed_from_summed_evidence(self) -> None:
        """Test prediction error is derived from the summed multi-channel evidence."""
        num_hyps = 2
        hypotheses = Hypotheses(
            evidence=np.array([5.0, 1.0]),  # hyp 0 is MLH
            locations=np.zeros((num_hyps, 3)),
            poses=np.tile(np.eye(3), (num_hyps, 1, 1)),
            possible=np.ones(num_hyps, dtype=bool),
        )

        evidence_by_channel = {
            "channel_a": np.array([1.5, 0.5]),
            "channel_b": np.array([0.5, -0.5]),
        }

        with patch.object(
            self.displacer,
            "_calculate_evidence_for_new_locations",
            side_effect=lambda **kw: evidence_by_channel[kw["input_channel"]],
        ):
            _, telemetry = self.displacer.displace_hypotheses_and_compute_evidence(
                displacements={
                    "channel_a": np.zeros(3),
                    "channel_b": np.zeros(3),
                },
                features={
                    "channel_a": {"pose_fully_defined": True},
                    "channel_b": {"pose_fully_defined": True},
                },
                evidence_update_threshold=-np.inf,
                graph_id="test_object",
                possible_hypotheses=hypotheses,
                total_hypotheses_count=num_hyps,
            )

        # MLH is index 0 (evidence 5.0), summed evidence at MLH = 1.5 + 0.5 = 2.0
        # With 2 channels (C=2), range is [-C, 2C] = [-2, 4], mapped to [0, 1]:
        # prediction_error = (-2.0 + 2*2) / (3*2) = 1/3
        self.assertAlmostEqual(telemetry.mlh_prediction_error, 1 / 3)
