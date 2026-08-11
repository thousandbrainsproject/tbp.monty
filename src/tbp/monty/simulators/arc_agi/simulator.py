# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from arcengine import GameAction

from arc_agi import Arcade, OperationMode
from tbp.monty.frameworks.environments.environment import SimulatedEnvironment

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class ArcAgiSimulator(SimulatedEnvironment):
    def __init__(
        self,
        game_id: str,
        data_path: str | Path | None = None,
    ):
        if data_path:
            self._arcade = Arcade(
                environments_dir=str(data_path),
                operation_mode=OperationMode.OFFLINE,
            )
        else:
            self._arcade = Arcade()

        self._env = self._arcade.make(game_id=game_id)

    def step(
        self,
        actions,  # noqa: ARG002
    ):
        return self._env.step(GameAction.ACTION1)

    def reset(self):
        pass
