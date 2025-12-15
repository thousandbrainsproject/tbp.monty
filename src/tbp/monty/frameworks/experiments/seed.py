# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import hashlib


def episode_seed(seed: int, epoch: int, episode: int) -> int:
    """Generate a seed for an episode.

    In some cases, for each episode, we want to deterministically modify
    the experiment's random seed based on the epoch and episode. We don't
    want to start with the same experiment random seed for each episode.
    For example, if we want to present objects in a random rotation for
    each episode, starting with the same random seed for each episode would
    result in the same rotation for each object in each episode.

    Args:
        seed: The experiment's random seed.
        epoch: The epoch number.
        episode: The episode number.

    Returns:
        A seed for the episode in the range [0, 2**32).
    """
    return (
        int(hashlib.sha256(f"{seed}-{epoch}-{episode}".encode()).hexdigest(), 16)
        % 2**32
    )
