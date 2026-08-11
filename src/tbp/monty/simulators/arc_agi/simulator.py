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
from typing import Callable, Sequence

from arc_agi import Arcade, OperationMode
from tbp.monty.frameworks.environments.environment import SimulatedEnvironment

__all__ = ["ArcAgiSimulator"]

from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_system_state import ProprioceptiveState
from tbp.monty.simulators.arc_agi.agents import ArcAgent

logger = logging.getLogger(__name__)


class ArcAgiSimulator(SimulatedEnvironment):
    """Expose live ARC games through Monty's frozen patch-sensor contract.

    Monty's ``reset`` and ``step`` only reset or move the patch sensor. Use
    ``reset_game`` and ``step_game`` to advance the underlying ARC game.
    """

    def __init__(
        self,
        game_id: str,
        agents: Sequence[Callable[[ArcAgiSimulator], ArcAgent]],
        data_path: str | Path | None = None,
    ) -> None:
        self.data_path = data_path
        if data_path:
            data_path = Path(data_path).expanduser()
            self._arcade = Arcade(
                environments_dir=str(data_path),
                operation_mode=OperationMode.OFFLINE,
            )
        else:
            self._arcade = Arcade(operation_mode=OperationMode.OFFLINE)

        self._agents = {}
        for agent_partial in agents:
            agent = agent_partial(self)
            self._agents[agent.id] = agent

        self.game_id = game_id
        self.env = self._arcade.make(game_id=game_id)
        assert self.env
        self._current_frame = self.env.observation_space

    @property
    def observations(self) -> Observations:
        obs = Observations()
        for agent in self._agents.values():
            obs[agent.id] = agent.observations
        return obs

    @property
    def states(self) -> ProprioceptiveState:
        states = ProprioceptiveState()
        for agent in self._agents.values():
            states[agent.id] = agent.state
        return states

    def step(self, actions) -> tuple[Observations, ProprioceptiveState]:
        for action in actions:
            agent = self._agents[action.agent_id]
            action.act(agent)

        return self.observations, self.states

    def reset(self):
        pass
