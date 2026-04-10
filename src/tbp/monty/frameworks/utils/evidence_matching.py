# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import numpy as np


def evidence_update_threshold(
    evidence_threshold_config: float | str,
    x_percent_threshold: float | str,
    max_global_evidence: float,
    evidence_all_channels: np.ndarray,
) -> float:
    """Determine how much evidence a hypothesis should have to be updated.

    Args:
        evidence_threshold_config: The heuristic for deciding which
            hypotheses should be updated.
        x_percent_threshold: The x_percent value to use for deciding
            on the `evidence_update_threshold` when the `x_percent_threshold` is
            used as a heuristic.
        max_global_evidence: Highest evidence of all hypotheses (i.e.,
            current mlh evidence),
        evidence_all_channels: Evidence values for all hypotheses.

    Returns:
        The evidence update threshold.

    Note:
        The logic of `evidence_threshold_config="all"` can be optimized by
        bypassing the `np.min` function here and bypassing the indexing of
        `np.where` function in the displacer. We want to update all the existing
        hypotheses, therefore there is no need to find the specific indices for
        them in the hypotheses space.

    Raises:
        InvalidEvidenceThresholdConfig: If `evidence_threshold_config` is
            not in the allowed values
    """
    # Return 0 for the threshold if there are no evidence scores
    if evidence_all_channels.size == 0:
        return 0.0

    if isinstance(evidence_threshold_config, (int, float)):
        return evidence_threshold_config

    if evidence_threshold_config == "mean":
        return np.mean(evidence_all_channels)

    if evidence_threshold_config == "median":
        return np.median(evidence_all_channels)

    if isinstance(
        evidence_threshold_config, str
    ) and evidence_threshold_config.endswith("%"):
        percentage_str = evidence_threshold_config.strip("%")
        percentage = float(percentage_str)
        assert percentage >= 0 and percentage <= 100, (
            "Percentage must be between 0 and 100"
        )
        x_percent_of_max = max_global_evidence * (percentage / 100)
        return max_global_evidence - x_percent_of_max

    if evidence_threshold_config == "x_percent_threshold":
        x_percent_of_max = max_global_evidence / 100 * float(x_percent_threshold)
        return max_global_evidence - x_percent_of_max

    if evidence_threshold_config == "all":
        return np.min(evidence_all_channels)

    raise InvalidEvidenceThresholdConfig(
        "evidence_threshold_config not in "
        "[int, float, '[int]%', 'mean', "
        "'median', 'all', 'x_percent_threshold']"
    )


class InvalidEvidenceThresholdConfig(ValueError):
    """Raised when the evidence update threshold is invalid."""

    pass
