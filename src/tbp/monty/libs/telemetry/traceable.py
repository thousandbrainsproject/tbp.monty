# Copyright 2025-2026 Thousand Brains Project
# Copyright 2021-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import functools

from .emitters import ContextualEmitter, Emitter

__all__ = ["Traceable"]


class Traceable:
    _EMITTER_CLASS = ContextualEmitter

    @property
    def telemetry_context(self) -> dict:
        return {}

    @functools.cached_property
    def tel(self) -> Emitter:
        return self._EMITTER_CLASS(self.telemetry_context)
