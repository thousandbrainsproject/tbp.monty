# Copyright 2025-2026 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import logging

from tbp.monty.frameworks.experiments.monty_experiment import (
    MontyExperiment,
)

__all__ = ["MontyObjectRecognitionExperiment"]

logger = logging.getLogger(__name__)


class MontyObjectRecognitionExperiment(MontyExperiment):
    """Experiment customized for object-pose recognition with a single object.

    Adds additional logging of the target object and pose for each episode and
    specific terminal states for object recognition. It also adds code for
    handling a matching and an exploration phase during each episode when training.

    Note that this experiment assumes a particular model configuration in order
    for the show_observations method to work: a zoomed-out "view_finder"
    RGBA sensor and an up-close "patch" depth sensor.
    """

    def pre_episode(self) -> None:
        super().pre_episode()

        # Pass the primary target object and the mapping from semantic IDs to labels
        # to the Monty model for logging and reporting evaluation results.
        if hasattr(self.env_interface, "semantic_id_to_label"):
            self.model.fixme_set_ground_truth(
                self.env_interface.primary_target,
                self.env_interface.semantic_id_to_label,
            )
        else:
            self.model.fixme_set_ground_truth(self.env_interface.primary_target)
