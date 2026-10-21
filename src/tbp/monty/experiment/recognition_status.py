# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "RecognitionConclusion",
    "RecognitionStatus",
]


class RecognitionConclusion(Enum):
    """Label for the terminal state of a Learning Module."""

    MATCH = "match"
    NO_MATCH = "no_match"
    TIME_OUT = "time_out"


@dataclass
class RecognitionStatus:
    """Recognition Status from each Learning Module."""

    conclusion: RecognitionConclusion | None = None
    telemetry: dict[str, Any] = field(default_factory=dict)
