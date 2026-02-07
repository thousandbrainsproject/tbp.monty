# Copyright 2025-2026 Thousand Brains Project
# Copyright 2021-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import asyncio
from dataclasses import dataclass

from tbp.monty.libs.telemetry import Event


@dataclass
class DispatcherConfig:
    pass


class Dispatcher:
    def __init__(self, sinks: list[Sink], rules: list[Rule]):
        self._sinks = sinks
        self._rules = rules

    async def dispatch(self, event: Event) -> None:
        targets = self._select_sinks(event)  # routing by labels/name
        # fanout concurrently, but bounded (semaphore) if needed
        await asyncio.gather(
            *(s.handle(event) for s in targets), return_exceptions=True
        )

    async def close(self) -> None:
        await asyncio.gather(*(s.close() for s in self._sinks), return_exceptions=True)
