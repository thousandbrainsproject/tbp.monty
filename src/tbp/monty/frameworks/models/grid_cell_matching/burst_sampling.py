# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from tbp.monty.frameworks.models.grid_cell_matching.cortical_scaffold import (
    CorticalScaffold,
)
from tbp.monty.frameworks.models.grid_cell_matching.grid_modules import GridModuleArray
from tbp.monty.frameworks.models.grid_cell_matching.hypothesis import (
    Hypothesis,
    HypothesisManager,
)

__all__ = ["GridCellBurstSampler", "BurstSamplingConfig"]

logger = logging.getLogger(__name__)


@dataclass
class BurstSamplingConfig:
    """Configuration for dynamic hypothesis management.

    Attributes:
        burst_trigger_slope: Trigger a sampling burst when the maximum
            evidence slope across all hypotheses drops below this value.
            Indicates the best hypothesis is not improving.
        deletion_trigger_slope: Delete hypotheses whose evidence slope is
            below this value. These hypotheses are stale.
        sampling_burst_duration: Number of steps during which new hypotheses
            are sampled after a burst is triggered.
        evidence_slope_window: Number of recent evidence values to use for
            computing the evidence slope.
        min_hypothesis_age: Minimum age (in matching steps) before a
            hypothesis can be deleted.
    """

    burst_trigger_slope: float = 0.0
    deletion_trigger_slope: float = -0.5
    sampling_burst_duration: int = 5
    evidence_slope_window: int = 10
    min_hypothesis_age: int = 5


class GridCellBurstSampler:
    """Dynamic hypothesis management via evidence slope tracking.

    P1 #5 FIX: The original implementation only pruned hypotheses below an
    evidence threshold but never sampled new ones. This meant the system
    could never recognise a second object after the first was identified.

    The burst sampler:
    1. Tracks per-hypothesis evidence slopes over a sliding window.
    2. Deletes hypotheses with declining slopes (stale, wrong object/pose).
    3. Triggers sampling bursts when the best hypothesis stops improving
       (global slope drops), indicating a possible object transition.
    4. During a burst, seeds new hypotheses from the current observation.
    """

    def __init__(self, config: BurstSamplingConfig | None = None):
        if config is None:
            config = BurstSamplingConfig()
        self.config = config
        self._burst_steps_remaining = 0
        self._in_burst = False

    def reset(self) -> None:
        """Reset burst sampler state for a new episode."""
        self._burst_steps_remaining = 0
        self._in_burst = False

    def step(
        self,
        hypotheses: list[Hypothesis],
        observation: dict,
        hypothesis_manager: HypothesisManager,
        cortical_scaffold: CorticalScaffold,
        grid_modules: GridModuleArray,
        known_object_ids: list[str],
    ) -> list[Hypothesis]:
        """Run one step of dynamic hypothesis management.

        1. Delete stale hypotheses.
        2. Check if a burst should be triggered.
        3. If in a burst, sample new hypotheses.

        Args:
            hypotheses: Current hypothesis list.
            observation: Current observation dict.
            hypothesis_manager: For creating new hypotheses.
            cortical_scaffold: Scaffold memory.
            grid_modules: Grid modules.
            known_object_ids: All known object IDs.

        Returns:
            Updated hypothesis list.
        """
        # 1. Delete stale hypotheses
        hypotheses = self._delete_stale(hypotheses)

        # 2. Check burst trigger
        if not self._in_burst:
            max_slope = self._get_max_slope(hypotheses)
            if max_slope < self.config.burst_trigger_slope and len(hypotheses) > 0:
                logger.info(
                    f"Burst triggered: max_slope={max_slope:.3f} < "
                    f"threshold={self.config.burst_trigger_slope}"
                )
                self._in_burst = True
                self._burst_steps_remaining = self.config.sampling_burst_duration

        # 3. Sample new hypotheses during burst
        if self._in_burst:
            new_hypotheses = hypothesis_manager.initialise_from_observation(
                observation, cortical_scaffold, grid_modules, known_object_ids
            )
            hypotheses.extend(new_hypotheses)
            logger.info(
                f"Burst sampling: added {len(new_hypotheses)} hypotheses "
                f"({self._burst_steps_remaining} steps remaining)"
            )

            self._burst_steps_remaining -= 1
            if self._burst_steps_remaining <= 0:
                self._in_burst = False

        return hypotheses

    def _delete_stale(
        self, hypotheses: list[Hypothesis]
    ) -> list[Hypothesis]:
        """Remove hypotheses with declining evidence slopes.

        Only hypotheses older than min_hypothesis_age are eligible for
        deletion.

        Args:
            hypotheses: Current hypothesis list.

        Returns:
            Filtered hypothesis list.
        """
        kept = []
        for hyp in hypotheses:
            if hyp.age < self.config.min_hypothesis_age:
                kept.append(hyp)
                continue
            slope = hyp.evidence_slope
            if slope >= self.config.deletion_trigger_slope:
                kept.append(hyp)
            else:
                logger.debug(
                    f"Deleting hypothesis {hyp.object_id} "
                    f"(slope={slope:.3f}, age={hyp.age})"
                )

        if len(kept) < len(hypotheses):
            logger.info(
                f"Deleted {len(hypotheses) - len(kept)} stale hypotheses "
                f"({len(kept)} remaining)"
            )
        return kept

    @staticmethod
    def _get_max_slope(hypotheses: list[Hypothesis]) -> float:
        """Get the maximum evidence slope across all hypotheses.

        Args:
            hypotheses: Current hypothesis list.

        Returns:
            Maximum slope, or -inf if no hypotheses.
        """
        if not hypotheses:
            return float("-inf")
        return max(h.evidence_slope for h in hypotheses)
