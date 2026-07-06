# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from tbp.monty.cmp import Message


def sm_percepts(percepts: Sequence[Message]) -> Sequence[Message]:
    """Return only the percepts sent by sensor modules.

    Args:
        percepts: Sequence of Message objects.

    Returns:
        The subset of percepts whose `sender_type` is "SM".
    """
    return [p for p in percepts if p.sender_type == "SM"]


def sm_location_mean(percepts: Sequence[Message]) -> npt.NDArray[np.float64]:
    """Compute the mean location across SM input channels.

    Args:
        percepts: Sequence of Message objects.

    Returns:
        The mean of the SM-channel locations.
    """
    return np.mean([p.location for p in sm_percepts(percepts)], axis=0)


def is_feature_step(percepts: Sequence[Message]) -> bool:
    """Return whether at least one SM channel delivered features.

    Distinguishes a feature step from a location-only step. The gate is computed
    from SM percepts only; LM output messages default `contains_features=True`, so
    an SM location-only step paired with an LM output must not count as a feature
    step.

    Args:
        percepts: Sequence of Message objects.

    Returns:
        True if any SM percept carries features.
    """
    return any(p.contains_features for p in sm_percepts(percepts))
