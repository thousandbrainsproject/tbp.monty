# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import copy
import logging
from typing import ClassVar

import numpy as np

from tbp.monty.frameworks.loggers.exp_logger import BaseMontyLogger
from tbp.monty.frameworks.loggers.graph_matching_loggers import (
    BasicGraphMatchingLogger,
    DetailedGraphMatchingLogger,
    SelectiveEvidenceLogger,
)
from tbp.monty.frameworks.models.evidence_matching.model import (
    MontyForEvidenceGraphMatching,
)

__all__ = ["MontyForGridCellMatching"]

logger = logging.getLogger(__name__)


class MontyForGridCellMatching(MontyForEvidenceGraphMatching):
    """Monty model for grid cell-based learning modules.

    Inherits from MontyForEvidenceGraphMatching to get the State-based
    vote combination logic (_combine_votes with sensor displacement
    transformation). The GridCellLM's send_out_vote produces the same
    format as the EvidenceGraphLM's: a dict with "possible_states" and
    "sensed_pose_rel_body".

    Overrides:
    - switch_to_exploratory_step: sets MLH evidence high enough for output
      during exploration.
    """

    LOGGING_REGISTRY: ClassVar[dict[str, type[BaseMontyLogger]]] = {
        "SILENT": BaseMontyLogger,
        "BASIC": BasicGraphMatchingLogger,
        "DETAILED": DetailedGraphMatchingLogger,
        "SELECTIVE": SelectiveEvidenceLogger,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def switch_to_exploratory_step(self):
        """Switch to exploratory step and ensure LM outputs during exploration.

        Sets MLH evidence above the object_evidence_threshold so the LM
        produces valid output States during exploration (needed for
        hierarchical connections and motor system).
        """
        super().switch_to_exploratory_step()
        for lm in self.learning_modules:
            if hasattr(lm, "current_mlh"):
                threshold = getattr(lm, "object_evidence_threshold", 1.0)
                lm.current_mlh["evidence"] = threshold + 1

    def _combine_votes(self, votes_per_lm):
        """Combine votes from grid cell LMs.

        Uses the MontyForEvidenceGraphMatching._combine_votes which
        handles State-based vote transformation with sensor displacement.
        The GridCellLM's send_out_vote returns the same format.

        Additionally supports the case where some LMs may use the older
        set-based voting (from GraphLM.send_out_vote), routing to the
        parent class.
        """
        # Check if any votes use the old set format
        has_set_votes = False
        has_dict_votes = False
        for v in votes_per_lm:
            if v is None:
                continue
            if isinstance(v, set):
                has_set_votes = True
            elif isinstance(v, dict):
                has_dict_votes = True

        if has_set_votes and not has_dict_votes:
            # All old-format: use base GraphLM voting
            from tbp.monty.frameworks.models.graph_matching import MontyForGraphMatching
            return MontyForGraphMatching._combine_votes(self, votes_per_lm)

        # Use evidence-style State-based voting
        return super()._combine_votes(votes_per_lm)
