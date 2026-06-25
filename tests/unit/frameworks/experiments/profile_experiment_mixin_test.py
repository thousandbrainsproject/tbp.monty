# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from unittest import TestCase

import pytest

from tbp.monty.frameworks.experiments.monty_experiment import MontyExperiment
from tbp.monty.frameworks.experiments.profile import (
    ProfileExperimentMixin,
)


class InheritanceProfileExperimentMixinTest(TestCase):
    @staticmethod
    def test_leftmost_subclassing_does_not_error() -> None:
        class GoodSubclass(ProfileExperimentMixin, MontyExperiment):
            pass

    @staticmethod
    def test_non_leftmost_subclassing_raises_error() -> None:
        with pytest.raises(TypeError):

            class BadSubclass(MontyExperiment, ProfileExperimentMixin):
                pass

    @staticmethod
    def test_missing_experiment_base_raises_error() -> None:
        with pytest.raises(TypeError):

            class BadSubclass(ProfileExperimentMixin):
                pass

    @staticmethod
    def test_experiment_subclasses_are_properly_detected() -> None:
        class SubExperiment(MontyExperiment):
            pass

        class Subclass(ProfileExperimentMixin, SubExperiment):
            pass
