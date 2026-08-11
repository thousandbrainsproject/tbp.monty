# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from arcengine import GameAction

from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.agents import AgentID


class GameReset(Action):
    arc_action = GameAction.RESET

    def act(self, actor):
        actor.actuate_game_reset(self)


class GameUp(Action):
    arc_action = GameAction.ACTION1

    def act(self, actor):
        actor.actuate_game_up(self)


class GameDown(Action):
    arc_action = GameAction.ACTION2

    def act(self, actor):
        actor.actuate_game_down(self)


class GameLeft(Action):
    arc_action = GameAction.ACTION3

    def act(self, actor):
        actor.actuate_game_left(self)


class GameRight(Action):
    arc_action = GameAction.ACTION4

    def act(self, actor):
        actor.actuate_game_right(self)


class GameUse(Action):
    arc_action = GameAction.ACTION5

    def act(self, actor):
        actor.actuate_game_use(self)


class GameClick(Action):
    arc_action = GameAction.ACTION6

    def __init__(self, agent_id: AgentID, x: int, y: int) -> None:
        super().__init__(agent_id)
        self.x = x
        self.y = y

    def act(self, actor):
        actor.actuate_game_click(self)


class GameUndo(Action):
    arc_action = GameAction.ACTION7

    def act(self, actor):
        actor.actuate_game_undo(self)
