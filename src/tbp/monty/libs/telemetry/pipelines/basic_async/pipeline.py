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

import asyncio
import queue
import sys
import threading
import time
from dataclasses import dataclass

from tbp.monty.libs.telemetry import Event

__all__ = ["BasicAsyncPipeline"]


_STOP = object()


@dataclass
class BasicAsyncPipelineConfig:
    max_queue_size: int = 10_000
    drain_timeout: float = 10.0


class BasicAsyncPipeline:
    def __init__(self, config: BasicAsyncPipelineConfig | None = None) -> None:
        self.config = config or BasicAsyncPipelineConfig()

        self._queue = queue.Queue(maxsize=self.config.max_queue_size)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._shutdown = threading.Event()

    def start(self) -> None:
        self._thread.start()

    def put(self, event: Event) -> None:
        if self._shutdown.is_set():
            self._log_stderr(str(event))
            return
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self._log_stderr("ERROR: telemetry pipeline queue is full!")

    def shutdown(self) -> None:
        self._log_stderr("WARNING: telemetry pipeline is shutting down!")
        self._log_stderr(
            "All newest events will be sent into /dev/stderr!"
        )  # in theory shouldn't be any, but just in case

        self._shutdown.set()
        self._queue.put(_STOP)
        self._queue.join()
        self._thread.join(self.config.drain_timeout)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._run_async_loop(loop))

    def _log_stderr(self, message: str) -> None:
        """Fallback to printing to stderr."""
        message_timestamped = f"{time.time()} | {message}"
        print(message_timestamped, file=sys.stderr)

    async def _run_async_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        dispatcher = 1  # TODO

        while True:
            event = await loop.run_in_executor(None, self._queue.get)
            try:
                if event is _STOP:
                    return

                _ = asyncio.create_task(dispatcher.dispatch(event))
            finally:
                self._queue.task_done()
