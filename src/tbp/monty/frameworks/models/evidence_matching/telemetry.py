# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from tbp.monty.cmp import AttentionRegion
    from tbp.monty.memento import Memento

__all__ = [
    "EvidenceGraphLMTelemetry",
    "EvidenceGraphLMTelemetryProtocol",
    "NoopEvidenceGraphLMTelemetry",
]


class EvidenceGraphLMTelemetryProtocol(Protocol):
    """What an :class:`EvidenceGraphLM` reports to its telemetry."""

    def reset(self) -> None: ...

    def attention_region(self, region: AttentionRegion | None) -> None: ...

    def state_dict(self) -> Memento: ...


class NoopEvidenceGraphLMTelemetry(EvidenceGraphLMTelemetryProtocol):
    """Records nothing; the default."""

    def reset(self) -> None:
        pass

    def attention_region(self, region: AttentionRegion | None) -> None:
        pass

    def state_dict(self) -> Memento:
        # The empty schema, so consumers indexing these keys stay simple.
        return dict(attention_regions=[])


class EvidenceGraphLMTelemetry(EvidenceGraphLMTelemetryProtocol):
    """Keeps what the LM proposed to the attention system each step.

    ``attention_regions`` holds one entry per
    :meth:`EvidenceGraphLM.propose_region` call: the region its proposers
    produced, or None when they produced nothing.
    """

    def __init__(self) -> None:
        self._attention_regions: list[AttentionRegion | None] = []

    def reset(self) -> None:
        self._attention_regions = []

    def attention_region(self, region: AttentionRegion | None) -> None:
        self._attention_regions.append(region)

    def state_dict(self) -> Memento:
        # Regions ride along as objects; BufferEncoder flattens them at
        # serialization time.
        return dict(attention_regions=list(self._attention_regions))
