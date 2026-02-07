# Copyright 2025-2026 Thousand Brains Project
# Copyright 2021-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import enum
import json
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Event"]


class BasicEvents(enum.Enum):
    LOG_DEBUG = "log.debug"
    LOG_INFO = "log.info"
    LOG_WARNING = "log.warning"
    LOG_ERROR = "log.error"


@dataclass(frozen=True)
class Event:
    name: str
    attrs: dict[str, str] | None = None
    payload: dict[str, Any] | None = None
    ts: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        try:
            return json.dumps(self.__dict__)
        except TypeError as e:  # foolproof
            fields = self.__dict__
            fields["payload"] = f"Failed to serialize: {e}"
            return json.dumps(fields)
