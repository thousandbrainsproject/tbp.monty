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

from .event import Event
from .pipelines import Pipeline

__all__ = ["Telemetry"]


class Telemetry:
    _instance: Telemetry | None = None

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    def emit(self, event: Event) -> None:
        self.pipeline.put(event)

    @classmethod
    def get(cls, pipeline: Pipeline | None = None) -> Telemetry:
        if not cls._instance:
            if not pipeline:
                raise RuntimeError("Attempted to use telemetry before initializing!")
            cls._instance = Telemetry(pipeline)

        return cls._instance

    @classmethod
    def shutdown(cls) -> None:
        if not cls._instance:
            raise RuntimeError("Attempted to shut down telemetry before initializing!")

        cls._instance.pipeline.shutdown()
