# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import json
import unittest

import numpy.testing as nptest

from tbp.monty.cmp import AttentionRegion
from tbp.monty.frameworks.models.buffer import BufferEncoder
from tbp.monty.frameworks.models.evidence_matching.telemetry import (
    EvidenceGraphLMTelemetry,
    NoopEvidenceGraphLMTelemetry,
)


def region(x: float) -> AttentionRegion:
    return AttentionRegion.uniform([[x, 0.0, 0.0]], 1.0, sender_id="learning_module_2")


class EvidenceGraphLMTelemetryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.telemetry = EvidenceGraphLMTelemetry()

    def test_starts_empty(self) -> None:
        self.assertEqual(self.telemetry.state_dict(), {"attention_regions": []})

    def test_records_each_proposal_in_order_including_none(self) -> None:
        self.telemetry.attention_region(region(1.0))
        self.telemetry.attention_region(None)
        self.telemetry.attention_region(region(3.0))

        proposed = self.telemetry.state_dict()["attention_regions"]

        self.assertEqual(len(proposed), 3)
        nptest.assert_array_equal(proposed[0].locations, [[1.0, 0.0, 0.0]])
        self.assertIsNone(proposed[1])
        nptest.assert_array_equal(proposed[2].locations, [[3.0, 0.0, 0.0]])

    def test_state_dict_is_a_snapshot(self) -> None:
        self.telemetry.attention_region(region(1.0))
        snapshot = self.telemetry.state_dict()["attention_regions"]

        self.telemetry.attention_region(region(2.0))

        self.assertEqual(len(snapshot), 1)

    def test_reset_discards_the_proposals(self) -> None:
        self.telemetry.attention_region(region(1.0))

        self.telemetry.reset()

        self.assertEqual(self.telemetry.state_dict(), {"attention_regions": []})

    def test_state_dict_serializes_with_the_buffer_encoder(self) -> None:
        self.telemetry.attention_region(region(1.0))
        self.telemetry.attention_region(None)

        encoded = json.loads(json.dumps(self.telemetry.state_dict(), cls=BufferEncoder))

        self.assertEqual(
            encoded["attention_regions"][0],
            {
                "locations": [[1.0, 0.0, 0.0]],
                "weights": [1.0],
                "sender_id": "learning_module_2",
                "inhibit_all": False,
            },
        )
        self.assertIsNone(encoded["attention_regions"][1])


class NoopEvidenceGraphLMTelemetryTest(unittest.TestCase):
    def test_records_nothing_and_keeps_the_schema(self) -> None:
        telemetry = NoopEvidenceGraphLMTelemetry()

        telemetry.attention_region(region(1.0))
        telemetry.reset()

        self.assertEqual(telemetry.state_dict(), {"attention_regions": []})
