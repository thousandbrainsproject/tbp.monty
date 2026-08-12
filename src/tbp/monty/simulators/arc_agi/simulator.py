# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# ruff: noqa: DOC201

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from arcengine import GameAction, GameState

from arc_agi import Arcade, EnvironmentWrapper, OperationMode
from tbp.monty.frameworks.environments.environment import (
    ObjectID,
    ObjectInfo,
    SemanticID,
    SimulatedObjectEnvironment,
)

__all__ = ["ArcAgiSimulator", "ArcOracleRegion"]

from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_system_state import ProprioceptiveState
from tbp.monty.math import QuaternionWXYZ, VectorXYZ
from tbp.monty.simulators.arc_agi.actions import GameReset
from tbp.monty.simulators.arc_agi.agents import ArcAgent

DISPLAY_SIZE = 64


@dataclass(frozen=True)
class ArcOracleRegion:
    """A fully visible sprite occurrence in a frozen ARC display frame."""

    region_id: str
    object_label: str
    display_origin: tuple[int, int]
    display_size: tuple[int, int]
    display_positions: tuple[tuple[int, int], ...]

    @property
    def bounding_box(self) -> tuple[int, int, int, int]:
        """Return the full display-space bounding box as ``x0, y0, x1, y1``."""
        x, y = self.display_origin
        width, height = self.display_size
        return x, y, x + width, y + height

    @property
    def global_anchor(self) -> tuple[float, float, float]:
        """Return the center of the full display-space bounding box."""
        x, y = self.display_origin
        width, height = self.display_size
        return x + width / 2, y + height / 2, 0.0

    @property
    def local_positions(self) -> tuple[tuple[int, int], ...]:
        """Return visible positions relative to the full sprite origin."""
        x, y = self.display_origin
        return tuple((px - x, py - y) for px, py in self.display_positions)


def _load_sprite_manifest(path: Path) -> dict[str, str]:
    """Index manifest source occurrences by their exact child label."""
    labels: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            for region in row["oracle_regions"]:
                labels[region["source_sprite_id"]] = row["object_label"]
    return labels


def _project_sprite(sprite, camera) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Project one sprite's visible pixels into display space."""
    pixels = np.asarray(sprite.render())
    pixels = np.where(pixels < 0, -1, pixels).astype(np.int8, copy=False)
    scale = min(DISPLAY_SIZE // camera.width, DISPLAY_SIZE // camera.height)
    x_offset = (DISPLAY_SIZE - camera.width * scale) // 2
    y_offset = (DISPLAY_SIZE - camera.height * scale) // 2
    origin = (
        x_offset + (sprite.x - camera.x) * scale,
        y_offset + (sprite.y - camera.y) * scale,
    )
    expanded = np.repeat(np.repeat(pixels, scale, axis=0), scale, axis=1)
    colors = np.full((DISPLAY_SIZE, DISPLAY_SIZE), -1, dtype=np.int8)
    mask = np.zeros((DISPLAY_SIZE, DISPLAY_SIZE), dtype=bool)
    x0, y0 = origin
    x1, y1 = x0 + expanded.shape[1], y0 + expanded.shape[0]
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(DISPLAY_SIZE, x1), min(DISPLAY_SIZE, y1)
    if dx0 < dx1 and dy0 < dy1:
        source = expanded[dy0 - y0 : dy1 - y0, dx0 - x0 : dx1 - x0]
        colors[dy0:dy1, dx0:dx1] = source
        mask[dy0:dy1, dx0:dx1] = source >= 0
    return mask, colors, origin


def _build_oracle_regions(
    *,
    game_id: str,
    game,
    frame: np.ndarray,
    labels: Mapping[str, str],
) -> tuple[ArcOracleRegion, ...]:
    """Build fully render-preserved regions for the current live level."""
    camera = game.camera
    frame = np.asarray(frame)

    sprites = game.current_level.get_sprites()
    projected = [_project_sprite(sprite, camera) for sprite in sprites]
    owner = np.full((DISPLAY_SIZE, DISPLAY_SIZE), -1, dtype=int)
    for index in sorted(range(len(sprites)), key=lambda item: sprites[item].layer):
        if sprites[index].is_visible:
            owner[projected[index][0]] = index

    regions = []
    for index, sprite in enumerate(sprites):
        region_id = f"{game_id}:{game.level_index + 1}:{index}:{sprite.name}"
        object_label = labels.get(region_id)
        if object_label is None or not sprite.is_visible:
            continue
        mask, colors, origin = projected[index]
        height, width = np.asarray(sprite.render()).shape
        scale = min(DISPLAY_SIZE // camera.width, DISPLAY_SIZE // camera.height)
        display_size = width * scale, height * scale
        x, y = origin
        if (
            not mask.any()
            or x < 0
            or y < 0
            or x + display_size[0] > DISPLAY_SIZE
            or y + display_size[1] > DISPLAY_SIZE
            or not np.all(owner[mask] == index)
            or not np.array_equal(frame[mask], colors[mask])
        ):
            continue
        rows, columns = np.nonzero(mask)
        positions = tuple((int(column), int(row)) for row, column in zip(rows, columns))
        regions.append(
            ArcOracleRegion(
                region_id=region_id,
                object_label=object_label,
                display_origin=origin,
                display_size=display_size,
                display_positions=positions,
            )
        )
    return tuple(regions)


class ArcAgiSimulator(SimulatedObjectEnvironment):
    """Simulator wrapper around the Arc AGI 3 game environment.

    This simulator responds to game-specific actions that advance the game, as well
    as SetSensorPose to move a sensor patch around the last frame received from
    the game.
    """

    def __init__(
        self,
        game_id: str,
        agents: Sequence[Callable[[ArcAgiSimulator], ArcAgent]],
        data_path: str | Path | None = None,
        sprite_manifest_path: str | Path | None = None,
    ) -> None:
        self._oracle_labels = (
            _load_sprite_manifest(Path(sprite_manifest_path).expanduser())
            if sprite_manifest_path
            else None
        )
        self._oracle_regions: tuple[ArcOracleRegion, ...] = ()
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
        self._env = self._arcade.make(game_id=game_id)
        self._refresh_oracle_regions()

    @property
    def env(self) -> EnvironmentWrapper:
        assert self._env
        return self._env

    @property
    def oracle_regions(self) -> tuple[ArcOracleRegion, ...]:
        """Return frozen privileged regions for the current reset frame."""
        return self._oracle_regions

    def get_oracle_region(self, region_id: str) -> ArcOracleRegion:
        """Return one frozen region or fail on a stale/unknown identifier.

        Raises:
            ValueError: If the region identifier is not in the frozen frame.
        """
        for region in self._oracle_regions:
            if region.region_id == region_id:
                return region
        raise ValueError(f"Unknown ARC oracle region {region_id!r}")

    def _refresh_oracle_regions(self) -> None:
        if self._oracle_labels is None:
            self._oracle_regions = ()
            return
        game = self.env._game
        frame = self.env.observation_space.frame[-1]
        self._oracle_regions = _build_oracle_regions(
            game_id=self.env.environment_info.game_id,
            game=game,
            frame=frame,
            labels=self._oracle_labels,
        )

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
        reset_requested = False
        for action in actions:
            agent = self._agents[action.agent_id]
            action.act(agent)
            reset_requested = reset_requested or isinstance(action, GameReset)

        game_state = self.env.observation_space.state

        # If the game ends up in a win or lose state, we want to
        # reset the game so that Monty can continue exploring
        # the space.
        if game_state is not GameState.NOT_FINISHED:
            self.env.step(GameAction.RESET)
            reset_requested = True

        if reset_requested:
            self._refresh_oracle_regions()

        return self.observations, self.states

    def add_object(
        self,
        name: str,
        position: VectorXYZ = (0.0, 0.0, 0.0),  # noqa: ARG002
        rotation: QuaternionWXYZ = (1.0, 0.0, 0.0, 0.0),  # noqa: ARG002
        scale: VectorXYZ = (1.0, 1.0, 1.0),  # noqa: ARG002
        semantic_id: SemanticID | None = None,  # noqa: ARG002
        primary_target_object: ObjectID | None = None,  # noqa: ARG002
    ) -> ObjectInfo:
        # We're hijacking the `add_object` method to switch games
        self.game_id = name
        self._env = self._arcade.make(game_id=name)
        self._refresh_oracle_regions()
        return ObjectInfo(semantic_id=SemanticID(0), object_id=ObjectID(0))

    def remove_all_objects(self) -> None:
        pass

    def reset(self):
        self.env.step(GameAction.RESET)
        self._refresh_oracle_regions()
        for agent in self._agents.values():
            agent.reset()
        return self.observations, self.states
