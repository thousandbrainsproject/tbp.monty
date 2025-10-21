# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class SalienceInput(Protocol):
    rgba: np.ndarray
    depth: np.ndarray


@dataclass
class RGBADepthObservation:
    rgba: np.ndarray
    depth: np.ndarray


class SalienceStrategy(Protocol):
    def __call__(self, obs: SalienceInput) -> np.ndarray: ...


class UniformSalienceStrategy(SalienceStrategy):
    def __call__(self, obs: SalienceInput) -> np.ndarray:
        return np.ones_like(obs.depth)
