# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

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

    @staticmethod
    def empty() -> Hypotheses:
        """Return a Hypotheses with all arrays shaped for zero hypotheses."""
        return Hypotheses(
            evidence=np.empty((0,), dtype=np.float64),
            locations=np.empty((0, 3), dtype=np.float64),
            poses=np.empty((0, 3, 3), dtype=np.float64),
            possible=np.empty((0,), dtype=np.bool_),
        )

    @staticmethod
    def concatenate(hyps: list[Hypotheses]) -> Hypotheses:
        """Concatenate multiple Hypotheses into a single unified Hypotheses.

        Returns:
            A single Hypotheses with all arrays concatenated.
        """
        return Hypotheses(
            evidence=np.hstack([h.evidence for h in hyps]),
            locations=np.vstack([h.locations for h in hyps]),
            poses=np.vstack([h.poses for h in hyps]),
            possible=np.hstack([h.possible for h in hyps]),
        )

    @property
    def count(self):
        """Return the number of hypotheses."""
        return self.evidence.shape[0]
