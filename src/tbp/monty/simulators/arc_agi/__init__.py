# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""ARC-AGI simulator integration."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .region_scan import ArcRegionScanPolicy, SetArcRegionPose
    from .simulator import ArcAgiSimulator, ArcOracleRegion

__all__ = [
    "ArcAgiSimulator",
    "ArcOracleRegion",
    "ArcRegionScanPolicy",
    "SetArcRegionPose",
]


def __getattr__(name: str) -> Any:
    if name in {"ArcRegionScanPolicy", "SetArcRegionPose"}:
        return getattr(import_module(f"{__name__}.region_scan"), name)
    if name in {"ArcAgiSimulator", "ArcOracleRegion"}:
        return getattr(import_module(f"{__name__}.simulator"), name)
    raise AttributeError(name)
