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


def location_only(percepts: Sequence[Message]) -> bool:
    """Return whether no SM channel delivered features.

    Distinguishes a location-only step from a feature step. The gate is computed
    from SM percepts only.

    Args:
        percepts: Sequence of Message objects.

    Returns:
        True if no SM percept carries features.
    """
    return not any(p.process_features_in_lm for p in sm_percepts(percepts))
