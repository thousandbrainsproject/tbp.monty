# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Any, Protocol

from tbp.monty.frameworks.environments.environment import SemanticID
from tbp.monty.frameworks.experiments.mode import ExperimentMode

__all__ = [
    "ExperimentMonty",
]


class ExperimentMonty(Protocol):
    """Experiment interface to Monty model."""

    def reset(self) -> None:
        """Reset the internal state of this Monty model."""
        ...

    def fixme_set_ground_truth(
        self,
        primary_target: dict[str, Any] | None = None,
        semantic_id_to_label: dict[SemanticID, str] | None = None,
    ) -> None:
        """Provide ground truth from experiment supervision.

        Args:
            primary_target: Optional primary target to recognize.
            semantic_id_to_label: Optional mapping from IDs to labels.
        """
        ...

    def update_ltm(self) -> None:
        """Transfer short-term buffer to long-term memory."""
        ...

    def set_experiment_mode(self, mode: ExperimentMode) -> None:
        """Set the experiment mode.

        Update state variables based on which method (train or evaluate) is being
        called at the experiment level.

        Args:
            mode: The experiment mode.
        """
        ...

    def is_done(self) -> bool:
        """Return `True` if the model has reached a terminal condition."""
        ...
