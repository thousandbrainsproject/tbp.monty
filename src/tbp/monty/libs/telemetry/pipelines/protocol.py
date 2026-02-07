# Copyright 2025-2026 Thousand Brains Project
# Copyright 2021-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Protocol

from tbp.monty.libs.telemetry import Event

__all__ = ["Pipeline"]


class Pipeline(Protocol):
    def put(self, event: Event) -> None: ...

    def shutdown(self) -> None: ...
