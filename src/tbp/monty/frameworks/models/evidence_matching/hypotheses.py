# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class Hypotheses:
    """Set of hypotheses consisting of evidence, locations, and poses.

    The three arrays are expected to have the same shape. Each index corresponds to a
    hypothesis.
    """

    evidence: npt.NDArray[np.float64]
    locations: npt.NDArray[np.float64]
    poses: npt.NDArray[np.float64]
    possible: npt.NDArray[np.bool_]


@dataclass
class ChannelHypotheses(Hypotheses):
    """A set of hypotheses for a single input channel."""

    input_channel: str
