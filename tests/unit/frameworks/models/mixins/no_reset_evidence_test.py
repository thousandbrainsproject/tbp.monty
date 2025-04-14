# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from unittest import TestCase

import pytest

from tbp.monty.frameworks.models.abstract_monty_classes import LearningModule
from tbp.monty.frameworks.models.evidence_matching import EvidenceGraphLM
from tbp.monty.frameworks.models.mixins.no_reset_evidence import (
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
