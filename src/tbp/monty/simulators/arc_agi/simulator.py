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
from pathlib import Path
from typing import Any

from arcengine import FrameDataRaw, GameAction

from arc_agi import Arcade, EnvironmentWrapper, OperationMode
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environments.arc import ArcFrameEnvironment
from tbp.monty.frameworks.sensors import SensorID

__all__ = ["ArcAgiSimulator"]

logger = logging.getLogger(__name__)


class ArcAgiSimulator(ArcFrameEnvironment):
    """Expose live ARC games through Monty's frozen patch-sensor contract.

    Monty's ``reset`` and ``step`` only reset or move the patch sensor. Use
    ``reset_game`` and ``step_game`` to advance the underlying ARC game.
    """

    def __init__(
        self,
        game_id: str,
        data_path: str | Path | None = None,
        seed: int = 0,
        agent_id: AgentID | str = "agent_id_0",
        sensor_id: SensorID | str = "patch",
        frame_size: int = 64,
        patch_size: int = 8,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            sensor_id=sensor_id,
            frame_size=frame_size,
            patch_size=patch_size,
        )
        self.data_path = None if data_path is None else Path(data_path).expanduser()
        if self.data_path is not None:
            self._arcade = Arcade(
                environments_dir=str(self.data_path),
                operation_mode=OperationMode.OFFLINE,
            )
        else:
            self._arcade = Arcade()

        self.seed = seed
        self.game_id = ""
        self._env: EnvironmentWrapper
        self.switch_to_game(game_id)

    @property
    def action_space(self) -> list[GameAction]:
        """Return the actions available in the current game state."""
        return self._env.action_space

    def switch_to_game(self, game_id: str, seed: int | None = None) -> FrameDataRaw:
        """Create and reset a game, then freeze its final response frame.

        Returns:
            The raw ARC reset response.

        Raises:
            ValueError: If ``game_id`` is empty.
            RuntimeError: If ARC-AGI cannot create the requested game.
        """
        if not game_id:
            raise ValueError("game_id must not be empty")

        next_seed = self.seed if seed is None else seed
        environment = self._arcade.make(game_id=game_id, seed=next_seed)
        if environment is None:
            raise RuntimeError(f"Could not create ARC environment {game_id}")
        response = environment.reset()
        if response is None:
            raise RuntimeError(f"ARC reset failed for {game_id}")

        self._set_frames(response.frame)
        self.game_id = game_id
        self.seed = next_seed
        self._env = environment
        self.frame_data = response
        return response

    def step_game(
        self,
        action: GameAction,
        data: dict[str, Any] | None = None,
        reasoning: dict[str, Any] | None = None,
    ) -> FrameDataRaw:
        """Apply one ARC game action and return its raw frame response.

        Returns:
            The updated raw ARC frame data.

        Raises:
            RuntimeError: If ARC-AGI fails to apply the action.
        """
        response = self._env.step(action, data=data, reasoning=reasoning)
        if response is None:
            raise RuntimeError(f"ARC step failed for {self.game_id}")
        self._set_frames(response.frame)
        self.frame_data = response
        return response

    def reset_game(self) -> FrameDataRaw:
        """Reset the current ARC game and return its raw frame response.

        Returns:
            The initial raw ARC frame data.

        Raises:
            RuntimeError: If ARC-AGI fails to reset the game.
        """
        response = self._env.reset()
        if response is None:
            raise RuntimeError(f"ARC reset failed for {self.game_id}")
        self._set_frames(response.frame)
        self.frame_data = response
        return response
