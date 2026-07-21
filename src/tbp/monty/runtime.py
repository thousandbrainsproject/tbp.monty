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

if TYPE_CHECKING:
    from tbp.monty.cmp import Message


def is_location_only_step(percepts: Sequence[Message]) -> bool:
    """Whether no sensor-module percept carries features this step.

    Args:
        percepts: Sequence of Message objects.

    Returns:
        True if none of the sensor-module percepts carry features.
    """
    return all(not p.process_features_in_lm for p in percepts if p.is_from_sm())
