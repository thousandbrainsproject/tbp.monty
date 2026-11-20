# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from dataclasses import dataclass

from tbp.monty.experiment.primary_target import PrimaryTarget


@dataclass
class LoggerArgs:
    episode_steps: int
    matching_steps: int
    total_train_steps: int
    train_episodes: int
    train_epochs: int
    total_eval_steps: int
    eval_episodes: int
    eval_epochs: int
    episode_seed: int
    target: PrimaryTarget | None
