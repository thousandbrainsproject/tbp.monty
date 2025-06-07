# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from dataclasses import dataclass

import numpy as np


@dataclass
class Hypotheses:
    """Set of hypotheses consisting of evidence, locations, and poses.

    The three arrays are expected to have the same shape. Each index corresponds to a
    hypothesis.
    """

    evidence: np.ndarray
    locations: np.ndarray
    poses: np.ndarray


@dataclass
class ChannelHypotheses(Hypotheses):
    """A set of hypotheses for a single input channel."""

    input_channel: str
