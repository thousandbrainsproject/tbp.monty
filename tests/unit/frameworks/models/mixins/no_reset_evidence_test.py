# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import json
from unittest import TestCase

import numpy as np
import pytest

from tbp.monty.frameworks.models.abstract_monty_classes import LearningModule
from tbp.monty.frameworks.models.buffer import BufferEncoder
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.frameworks.models.mixins.no_reset_evidence import (
    HypothesesUpdaterChannelTelemetry,
    TheoreticalLimitLMLoggingMixin,
)


class InheritanceTheoreticalLMLoggingMixinTest(TestCase):
    @staticmethod
    def test_mixin_used_with_compatible_learning_module_does_not_error() -> None:
        class Compatible(TheoreticalLimitLMLoggingMixin, EvidenceGraphLM):
            pass

    @staticmethod
    def test_mixin_used_with_non_compatible_learning_module_raises_error() -> None:
        with pytest.raises(TypeError):

            class NonCompatible(TheoreticalLimitLMLoggingMixin, LearningModule):
                pass


class HypothesesUpdaterChannelTelemetryTest(TestCase):
    def test_buffer_encoder_encodes_hypotheses_updater_channel_telemetry(self) -> None:
        telemetry = HypothesesUpdaterChannelTelemetry(
            hypotheses_updater={},
            evidence=np.array([0, 1]),
            rotations=np.array([0, 1]),
            pose_errors=np.array([0, 1]),
        )
        encoded = json.loads(json.dumps(telemetry, cls=BufferEncoder))
        self.assertEqual(
            encoded,
            {
                "hypotheses_updater": {},
                "evidence": [0, 1],
                "rotations": [0, 1],
                "pose_errors": [0, 1],
            },
        )
