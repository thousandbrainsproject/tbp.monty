# Copyright 2025-2026 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Sequence, TypedDict, cast

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation
from typing_extensions import NotRequired

from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.experiments.seed import episode_seed
from tbp.monty.frameworks.utils.transform_utils import (
    rotation_as_quat,
)
from tbp.monty.math import EulerAnglesXYZ, QuaternionWXYZ, VectorXYZ


class MultiObjectNames(TypedDict):
    targets_list: Sequence[str]
    source_object_list: Sequence[str]
    num_distractors: int


class ObjectInitParams(TypedDict):
    position: VectorXYZ
    rotation: QuaternionWXYZ
    scale: VectorXYZ
    euler_rotation: npt.NDArray[np.float64] | EulerAnglesXYZ
    quat_rotation: NotRequired[npt.NDArray[np.float64]]


class Default:
    def __call__(
        self,
        seed: int,
        mode: ExperimentMode,
        epoch: int,  # noqa: ARG002
        episode: int,
    ) -> ObjectInitParams:
        seed = episode_seed(seed, mode, episode)
        rng = np.random.RandomState(seed)
        euler_rotation = rng.uniform(0, 360, 3)
        q = Rotation.from_euler("xyz", euler_rotation, degrees=True)
        return dict(
            rotation=cast("QuaternionWXYZ", tuple(rotation_as_quat(q))),
            euler_rotation=euler_rotation,
            position=(rng.uniform(-0.5, 0.5), 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
        )


class Predefined(Default):
    def __init__(
        self,
        positions: Sequence[VectorXYZ] | None = None,
        rotations: Sequence[EulerAnglesXYZ] | None = None,
        scales: Sequence[VectorXYZ] | None = None,
        change_every_episode: bool | None = None,
    ):
        # NOTE: added param change_every_episode. This is so if I want to run an
        # experiment and specify an exact list of objects, with specific poses per
        # object, I can set this to True. Otherwise, I have to loop over all objects
        # for every pose specified.
        self.positions = positions or [(0.0, 1.5, 0.0)]
        self.rotations = rotations or [(0.0, 0.0, 0.0), (45.0, 0.0, 0.0)]
        self.scales = scales or [(1.0, 1.0, 1.0)]
        self.change_every_episode = change_every_episode

    def __call__(
        self,
        seed: int,  # noqa: ARG002
        mode: ExperimentMode,  # noqa: ARG002
        epoch: int,
        episode: int,
    ) -> ObjectInitParams:
        mod_counter = episode if self.change_every_episode else epoch
        q = Rotation.from_euler(
            "xyz",
            self.rotations[mod_counter % len(self.rotations)],
            degrees=True,
        )
        return dict(
            rotation=cast("QuaternionWXYZ", tuple(rotation_as_quat(q))),
            euler_rotation=self.rotations[mod_counter % len(self.rotations)],
            quat_rotation=q.as_quat(),
            position=self.positions[mod_counter % len(self.positions)],
            scale=self.scales[mod_counter % len(self.scales)],
        )

    def __len__(self):
        return len(self.all_combinations_of_params())

    def all_combinations_of_params(self):
        param_list = []
        for i in range(len(self.rotations)):
            for j in range(len(self.scales)):
                for k in range(len(self.positions)):
                    params = dict(
                        rotations=[self.rotations[i]],
                        scales=[self.scales[j]],
                        positions=[self.positions[k]],
                    )
                    param_list.append(params)
        return param_list


class RandomRotation(Default):
    def __init__(
        self,
        position: VectorXYZ | None = None,
        scale: VectorXYZ | None = None,
    ):
        if position is not None:
            self.position = position
        else:
            self.position = (0.0, 1.5, 0.0)
        if scale is not None:
            self.scale = scale
        else:
            self.scale = (1.0, 1.0, 1.0)

    def __call__(
        self,
        seed: int,
        mode: ExperimentMode,
        epoch: int,  # noqa: ARG002
        episode: int,
    ) -> ObjectInitParams:
        seed = episode_seed(seed, mode, episode)
        rng = np.random.RandomState(seed)
        euler_rotation = rng.uniform(0, 360, 3)
        q = Rotation.from_euler("xyz", euler_rotation, degrees=True)
        return dict(
            rotation=cast("QuaternionWXYZ", tuple(rotation_as_quat(q))),
            euler_rotation=euler_rotation,
            quat_rotation=q.as_quat(),
            position=self.position,
            scale=self.scale,
        )
