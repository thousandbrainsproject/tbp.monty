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

from tbp.monty.frameworks.models.evidence_matching.region_proposal.inhibit_all_on_recognition import (  # noqa: E501
    InhibitAllOnRecognition,
)
from tbp.monty.frameworks.models.evidence_matching.region_proposal.protocol import (
    RegionContext,
)


def context_recognizing(object_id: str | None) -> MagicMock:
    # The proposer reads nothing else off the context.
    context = MagicMock(spec=RegionContext)
    context.recognized_object = object_id
    return context


class InhibitAllOnRecognitionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.proposer = InhibitAllOnRecognition()

    def test_proposes_nothing_while_no_object_is_recognized(self) -> None:
        self.assertIsNone(self.proposer(context_recognizing(None)))

    def test_signals_inhibit_all_with_no_locations_on_recognition(self) -> None:
        region = self.proposer(context_recognizing("mug"))

        self.assertTrue(region.inhibit_all)
        self.assertEqual(len(region), 0)

    def test_signals_once_per_recognized_object(self) -> None:
        self.proposer(context_recognizing("mug"))

        self.assertIsNone(self.proposer(context_recognizing("mug")))

    def test_signals_again_for_another_object(self) -> None:
        self.proposer(context_recognizing("mug"))

        self.assertIsNotNone(self.proposer(context_recognizing("bowl")))

    def test_losing_and_regaining_recognition_does_not_signal_again(self) -> None:
        self.proposer(context_recognizing("mug"))
        self.proposer(context_recognizing(None))

        self.assertIsNone(self.proposer(context_recognizing("mug")))

    def test_reset_lets_an_object_signal_again(self) -> None:
        self.proposer(context_recognizing("mug"))
        self.proposer.reset()

        self.assertIsNotNone(self.proposer(context_recognizing("mug")))
