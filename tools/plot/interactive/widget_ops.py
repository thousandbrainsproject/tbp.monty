# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Callable, Iterable, Protocol, runtime_checkable

from tools.plot.interactive.generics import S, W
from tools.plot.interactive.topics import TopicMessage
from tools.plot.interactive.widget_updaters import WidgetUpdaterProto

# @runtime_checkable
# class WidgetOpsProto(Protocol[W, S]):
#     pass


@runtime_checkable
class WidgetOpsProto(Protocol):
    pass


@runtime_checkable
class SupportsAdd(Protocol[W]):
    def add(self, callback: Callable[[W, str], None]) -> W: ...


@runtime_checkable
class SupportsRemove(Protocol[W]):
    def remove(self, widget: W) -> None: ...


@runtime_checkable
class SupportsExtractState(Protocol[W, S]):
    def extract_state(self, widget: W | None) -> S | None: ...


@runtime_checkable
class SupportsSetState(Protocol[W, S]):
    def set_state(self, widget: W | None, value: S | None) -> None: ...


@runtime_checkable
class HasStateToMessages(Protocol[S]):
    def state_to_messages(self, state: S | None) -> Iterable[TopicMessage]: ...


@runtime_checkable
class HasUpdaters(Protocol[W]):
    updaters: Iterable[WidgetUpdaterProto[W]]
