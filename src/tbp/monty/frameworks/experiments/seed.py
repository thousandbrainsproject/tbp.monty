# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import hashlib

from tbp.monty.frameworks.experiments.mode import ExperimentMode

__all__ = ["episode_seed"]


def episode_seed(seed: int, mode: ExperimentMode, episode: int) -> int:
    """Generate a seed for an episode.

    In some cases, for each episode, we want to deterministically modify
    the experiment's random seed based on the experiment mode and episode.
    We don't want to start with the same experiment random seed for each episode.
    For example, if we want to present objects in a random rotation for
    each episode, starting with the same random seed for each episode would
    result in the same rotation for each object in each episode. As another
    example, if we add noise during training, we want to add different noise
    during evaluation, even if the episode is the same.

    Args:
        seed: The experiment's random seed.
        mode: The experiment mode.
        episode: The episode number.

    Returns:
        A seed for the episode in the range [0, 2**32).
    """
    return (
        int(
            hashlib.sha256(f"{seed}-{mode.value}-{episode}".encode()).hexdigest(),
            16,
        )
        % 2**32
    )
