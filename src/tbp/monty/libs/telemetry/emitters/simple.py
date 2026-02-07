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

from typing import Any

from tbp.monty.libs.telemetry import BasicEvents, Event, Telemetry

__all__ = ["SimpleEmitter"]


class SimpleEmitter:
    def __init__(self, default_attributes: dict) -> None:
        self.telemetry = Telemetry.get()
        self.default_attributes = default_attributes

    def emit(
        self,
        name: str,
        attrs: dict[str, str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        attrs_merged = {**self.default_attributes, **(attrs or {})}

        event = Event(name, attrs_merged, payload)
        self.telemetry.emit(event)

    def _log(
        self,
        event_name: BasicEvents,
        message: str,
        attrs: dict[str, str] | None = None,
        payload: dict[str, Any] | None = None,
    ):
        log_payload = {"message": message, **(payload or {})}

        event = Event(str(event_name.value), attrs=attrs, payload=log_payload)
        self.telemetry.emit(event)

    def debug(self, message: str, **kwargs) -> None:
        self._log(BasicEvents.LOG_DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        self._log(BasicEvents.LOG_INFO, message, **kwargs)

    def warn(self, message: str, **kwargs) -> None:
        self._log(BasicEvents.LOG_WARNING, message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        self._log(BasicEvents.LOG_ERROR, message, **kwargs)
