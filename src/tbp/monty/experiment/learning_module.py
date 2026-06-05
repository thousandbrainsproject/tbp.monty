# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Protocol

from tbp.monty.frameworks.experiments.mode import ExperimentMode

__all__ = [
    "ExperimentLearningModule",
]


class ExperimentLearningModule(Protocol):
    """Experiment interface to a Learning Module."""

    def reset_stm(self) -> None:
        """Reset short-term memory buffer.

        Do things like reset buffers or possible_matches before training.
        """
        ...

    def fixme_reset_ground_truth(self, primary_target=None) -> None:
        """Reset internal state based on ground truth.

        Args:
            primary_target: The primary target for the learning module to recognize.
        """
        ...

    def update_ltm_from_stm(self) -> None:
        """Update long-term memory from short-term memory buffer."""
        ...

    def fixme_update_ground_truth(self) -> None:
        """Update internal state based on ground truth."""
        ...

    def set_experiment_mode(self, mode: ExperimentMode) -> None:
        """Set the experiment mode.

        Update state variables based on which method (train or evaluate) is being called
        at the experiment level.

        Args:
            mode: The experiment mode.
        """
        ...
