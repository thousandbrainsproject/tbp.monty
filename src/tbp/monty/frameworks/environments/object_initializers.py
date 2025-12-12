# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import numpy as np
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.experiments.seed import episode_seed
from tbp.monty.frameworks.utils.transform_utils import scipy_to_numpy_quat


class Default:
    def __call__(self, seed: int, epoch: int, episode: int):
        seed = episode_seed(seed, epoch, episode)
        rng = np.random.RandomState(seed)
        euler_rotation = rng.uniform(0, 360, 3)
        q = Rotation.from_euler("xyz", euler_rotation, degrees=True).as_quat()
        quat_rotation = scipy_to_numpy_quat(q)
        return dict(
            rotation=quat_rotation,
            euler_rotation=euler_rotation,
            position=(rng.uniform(-0.5, 0.5), 0.0, 0.0),
            scale=[1.0, 1.0, 1.0],
        )

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash(self.__dict__)


class Predefined(Default):
    def __init__(
        self, positions=None, rotations=None, scales=None, change_every_episode=None
    ):
        # NOTE: added param change_every_episode. This is so if I want to run an
        # experiment and specify an exact list of objects, with specific poses per
        # object, I can set this to True. Otherwise I have to loop over all objects
        # for every pose specified.
        self.positions = positions or [[0.0, 1.5, 0.0]]
        self.rotations = rotations or [[0.0, 0.0, 0.0], [45.0, 0.0, 0.0]]
        self.scales = scales or [[1.0, 1.0, 1.0]]
        self.change_every_episode = change_every_episode

    def __call__(self, _: int, epoch: int, episode: int):
        mod_counter = episode if self.change_every_episode else epoch
        q = Rotation.from_euler(
            "xyz",
            self.rotations[mod_counter % len(self.rotations)],
            degrees=True,
        ).as_quat()
        quat_rotation = scipy_to_numpy_quat(q)
        return dict(
            rotation=quat_rotation,
            euler_rotation=list(self.rotations[mod_counter % len(self.rotations)]),
            quat_rotation=q,
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
    def __init__(self, position=None, scale=None):
        if position is not None:
            self.position = position
        else:
            self.position = [0.0, 1.5, 0.0]
        if scale is not None:
            self.scale = scale
        else:
            self.scale = [1.0, 1.0, 1.0]

    def __call__(self, seed: int, epoch: int, episode: int):
        seed = episode_seed(seed, epoch, episode)
        rng = np.random.RandomState(seed)
        euler_rotation = rng.uniform(0, 360, 3)
        q = Rotation.from_euler("xyz", euler_rotation, degrees=True).as_quat()
        quat_rotation = scipy_to_numpy_quat(q)
        return dict(
            rotation=quat_rotation,
            euler_rotation=euler_rotation,
            quat_rotation=q,
            position=self.position,
            scale=self.scale,
        )
