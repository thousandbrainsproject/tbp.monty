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

from typing import Any, Protocol

__all__ = ["Emitter"]


class Emitter(Protocol):
    def emit(
        self,
        name: str,
        attrs: dict[str, str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None: ...

    def log(self, level: str, message: str) -> None: ...
    def info(self, message: str) -> None: ...
    def warn(self, message: str) -> None: ...
    def error(self, message: str) -> None: ...
