# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from enum import Enum

__all__ = ["ExperimentMode"]


class ExperimentMode(Enum):
    """Experiment mode."""

    EVAL = "eval"
    """Evaluation mode."""
    TRAIN = "train"
    """Training mode."""
