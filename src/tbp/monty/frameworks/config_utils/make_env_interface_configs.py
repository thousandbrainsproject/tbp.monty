# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.utils.transform_utils import scipy_to_numpy_quat

if TYPE_CHECKING:
    from numbers import Number


class DefaultObjectInitializer:
    def __call__(self):
        euler_rotation = self.rng.uniform(0, 360, 3)
        q = Rotation.from_euler("xyz", euler_rotation, degrees=True).as_quat()
        quat_rotation = scipy_to_numpy_quat(q)
        return dict(
            rotation=quat_rotation,
            euler_rotation=euler_rotation,
            position=(self.rng.uniform(-0.5, 0.5), 0.0, 0.0),
            scale=[1.0, 1.0, 1.0],
        )

    def post_epoch(self):
        pass

    def post_episode(self):
        pass

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash(self.__dict__)


class PredefinedObjectInitializer(DefaultObjectInitializer):
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
        self.current_epoch = 0
        self.current_episode = 0
        self.change_every_episode = change_every_episode

    def __call__(self):
        mod_counter = (
            self.current_episode if self.change_every_episode else self.current_epoch
        )
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

    def __repr__(self):
        string = "PredefinedObjectInitializer with params: \n"
        string += f"\t positions: {self.positions}\n"
        string += f"\t rotations: {self.rotations}\n"
        string += f"\t change every episode: {self.change_every_episode}"
        return string

    def __len__(self):
        return len(self.all_combinations_of_params())

    def post_epoch(self):
        self.current_epoch += 1

    def post_episode(self):
        self.current_episode += 1

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


class RandomRotationObjectInitializer(DefaultObjectInitializer):
    def __init__(self, position=None, scale=None):
        if position is not None:
            self.position = position
        else:
            self.position = [0.0, 1.5, 0.0]
        if scale is not None:
            self.scale = scale
        else:
            self.scale = [1.0, 1.0, 1.0]

    def __call__(self):
        euler_rotation = self.rng.uniform(0, 360, 3)
        q = Rotation.from_euler("xyz", euler_rotation, degrees=True).as_quat()
        quat_rotation = scipy_to_numpy_quat(q)
        return dict(
            rotation=quat_rotation,
            euler_rotation=euler_rotation,
            quat_rotation=q,
            position=self.position,
            scale=self.scale,
        )


"""
Utilities for generating multi-LM environment interface args.
"""


def make_sensor_positions_on_grid(
    n_sensors: int,
    delta: Number = 0.01,
    order_by: str = "distance",
    add_view_finder: bool = True,
) -> np.ndarray:
    """Generate sensor positions on a 2D grid.

    Create mounting positions for an arbitrary number of sensors, where the
    sensors lie on an imaginary grid on the xy plane (and z = 0). Sensor position
    0 is always centered at (0, 0, 0), and all other sensors are clustered
    around it. The method for selecting which grid points around the center to assign
    each sensor is determined by the `order_by` argument (see below).

    By default, `n_sensors + 1` positions are returned; the first `n_sensors` positions
    are for regular sensors, and an additional position is appended by default to
    accommodate a view finder. The view finder position (if used) is the same as
    sensor position 0 (i.e., (0, 0, 0)).

    Args:
        n_sensors: Number of sensors. Count should not include a view finder.
        delta: The grid spacing length. By default, sensors will be
            placed every centimeter (units are in meters).
        order_by: How to select points on the grid that will contain
            sensors.
             - "spiral": sensors are numbered along a counter-clockwise spiral
                spreading outwards from the center.
             - "distance": sensors are ordered by their distance from the center.
                This can result in a more jagged pattern along the edges but
                results in sensors generally more packed towards the center.
                Positions that are equidistant from the center are ordered
                counterclockwise starting at 3 o'clock.
        add_view_finder: Whether to include an extra position module
            at the origin to serve as a view finder. Defaults to `True`.

    Returns:
        A 2D array of sensor positions where each row is an array of (x, y, z)
        positions. If `add_view_finder` is True, the array has `n_sensors + 1` rows,
        where the last row corresponds to the view finder's position and is identical to
        row 0. Otherwise, the array has `n_sensors` rows. row 0 is always centered at
        (0, 0, 0), and all other rows are offset relative to it.

    """
    assert n_sensors > 0, "n_sensors must be greater than 0"
    assert delta > 0, "delta must be greater than 0"
    assert order_by in ["spiral", "distance"], "order_by must be 'spiral' or 'distance'"

    # Find smallest square grid size that can fit n_lms with odd-length sides.
    grid_size = 1
    while n_sensors > grid_size**2:
        grid_size += 2

    # Make coordinate grids, where the center is (0, 0).
    points = np.arange(-grid_size // 2 + 1, grid_size // 2 + 1)
    x, y = np.meshgrid(points, points)
    y = np.flipud(y)  # Flip y-axis to match habitat coordinate system (positive is up).
    i_mid = grid_size // 2

    if order_by == "distance":
        dists = x**2 + y**2
        unique_dists = np.sort(np.unique(dists))
        assert unique_dists[0] == 0
        indices = []
        for i in range(len(unique_dists)):
            u = unique_dists[i]
            inds = np.argwhere(dists == u)
            angles = np.arctan2(i_mid - inds[:, 1], inds[:, 0] - i_mid)
            sorting_inds = np.argsort(angles)
            inds = inds[sorting_inds]
            indices.extend(list(inds))

    elif order_by == "spiral":
        indices = [(i_mid, i_mid)]

        # Directions for moving in spiral: right, down, left, up
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        current_dir = 0  # Start moving right
        steps = 1  # How many steps to take in current direction
        steps_taken = 0  # Steps taken in current direction
        row, col = i_mid, i_mid  # Start at center

        # Generate spiral pattern until we have enough points
        while len(indices) < grid_size**2:
            # Move in current direction
            row += directions[current_dir][0]
            col += directions[current_dir][1]

            # Add point if it's within bounds
            if 0 <= row < grid_size and 0 <= col < grid_size:
                indices.append((row, col))

            steps_taken += 1

            # Check if we need to change direction
            if steps_taken == steps:
                steps_taken = 0
                current_dir = (current_dir + 1) % 4
                # Increase steps every 2 direction changes (completing half circle)
                if current_dir % 2 == 0:
                    steps += 1

    indices = np.array(indices)[:n_sensors]

    # Convert indices to locations in agent space.
    positions = []
    for idx in indices:
        positions.append((x[idx[0], idx[1]] * delta, y[idx[0], idx[1]] * delta))
    positions = np.array(positions)

    # Add z-positions.
    positions = np.hstack((positions, np.zeros([positions.shape[0], 1])))

    # Optionally append entry for a view finder which is a duplicate of row zero.
    # Should be (0, 0, 0).
    if add_view_finder:
        positions = np.vstack([positions, positions[0].reshape(1, -1)])

    return positions
