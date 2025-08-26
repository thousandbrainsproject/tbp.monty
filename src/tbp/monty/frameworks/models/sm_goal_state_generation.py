# Copyright 2025 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import quaternion
from matplotlib import colors
from numpy.typing import ArrayLike
from scipy import ndimage
from scipy.spatial.transform import Rotation
from skimage import measure

from tbp.monty.frameworks.models.abstract_monty_classes import SensorModule
from tbp.monty.frameworks.models.goal_state_generation import SmGoalStateGenerator
from tbp.monty.frameworks.models.states import GoalState

logger = logging.getLogger(__name__)




class TargetFindingGsg(SmGoalStateGenerator):
    """Sensor module goal-state generator that finds targets in the image."""

    def __init__(
        self,
        parent_sm: SensorModule,
        goal_tolerances: dict | None = None,
        save_telemetry: bool = False,
        match: ArrayLike = "red",
        threshold: float = 0.5,
        min_size: int = 0,
        **kwargs,
    ) -> None:
        """Initialize the GSG.

        Args:
            parent_sm: The sensor module class instance that the GSG is embedded
                within.
            goal_tolerances: The tolerances for each attribute of the goal-state
                that can be used by the GSG when determining whether a goal-state is
                achieved.
            save_telemetry: Whether to save telemetry data.
            **kwargs: Additional keyword arguments. Unused.
        """
        super().__init__(parent_sm, goal_tolerances, save_telemetry, **kwargs)
        self.target_finder = TargetFinder(match, threshold, min_size)
        self.decay_field = DecayField()

    def _generate_output_goal_state(
        self,
        raw_observation: dict | None = None,
        processed_observation: dict | None = None,
    ) -> list[GoalState]:
        """Generate the output goal state(s).

        Generates the output goal state(s) based on the driving goal state and the
        achieved goal state(s).

        Args:
            raw_observation: The parent sensor module's raw observations.
            processed_observation: The parent sensor module's processed observations.

        Returns:
            The output goal state(s).
        """
        targets = self.target_finder(raw_observation["rgba"])

        # Get coordinates of image data in (ypix, xpix, vector3d) format.
        n_rows, n_cols = raw_observation["rgba"].shape[0:2]
        pos_2d = raw_observation["semantic_3d"][:, 0:3].reshape(n_rows, n_cols, 3)

        # Make goal states for each target.
        goal_states = []
        for t in targets:
            target_loc = pos_2d[t["center_pix"][0], t["center_pix"][1]]
            goal_states.append(self._create_goal_state(target_loc))

        # Update the decay field with the current sensed location.
        cur_loc = pos_2d[n_rows//2, n_cols//2]
        self.decay_field.add(cur_loc)

        # Modify goal-state confidence values based on the decay field.
        for g in goal_states:
            val = self.decay_field(g.location)
            g.confidence = val

        # Step the decay field.
        self.decay_field.step()

        # Return the goal states.
        return goal_states

    def _create_goal_state(
        self,
        location: np.ndarray,
        morphological_features: Optional[Dict[str, Any]] = None,
        non_morphological_features: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
        use_state: bool = True,
        goal_tolerances: Optional[Dict[str, Any]] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> GoalState:
        """Create a goal state with default values."""
        return GoalState(
            location=location,
            morphological_features=morphological_features,
            non_morphological_features=non_morphological_features,
            confidence=confidence,
            use_state=use_state,
            sender_id=self.parent_sm.sensor_module_id,
            sender_type="GSG",
            goal_tolerances=goal_tolerances,
            info=info,
        )


class TargetFinder:
    def __init__(
        self, match: ArrayLike = "red", threshold: float = 0.5, min_size: int = 0
    ):
        self._match = np.array(colors.to_rgb(match))
        self._threshold = threshold
        self._min_size = min_size

    def __call__(self, image: np.ndarray) -> list[dict]:
        """Find targets.

        Args:
            image: numpy array of shape (H, W, C) with RGB values

        Returns:
            list of dicts with keys "center_pix" and "size_pix"
        """
        # Drop alpha if present
        rgb = image[:, :, :3]
        rgb = rgb / 255.0 if np.issubdtype(rgb.dtype, np.integer) else rgb

        # Compute euclidean distance between each pixel's RGB values and target color
        distances = np.sqrt(np.sum((rgb - self._match)**2, axis=2))

        # Create binary mask for matching pixels
        mask = distances < self._threshold

        # Remove small noise using morphological operations
        # Use a small disk structure to clean up the mask
        # structure = ndimage.generate_binary_structure(2, 2)
        # red_mask = ndimage.binary_opening(red_mask, structure=structure, iterations=1)
        # red_mask = ndimage.binary_closing(red_mask, structure=structure, iterations=1)
        # Label connected components
        labeled_image, num_features = ndimage.label(mask)

        # Find center of each connected component
        targets = []
        for i in range(1, num_features + 1):
            # Get pixels belonging to this component
            component_mask = labeled_image == i

            # Skip if component is too small
            if (self._min_size > 0) and (np.sum(component_mask) < self._min_size):
                continue

            # Calculate center of mass
            center_y, center_x = ndimage.center_of_mass(component_mask)
            row, col = round(center_y), round(center_x)
            targets.append(
                {
                    "center_pix": (row, col),
                    "size_pix": round(np.sum(component_mask)),
                }
            )

        return targets

"""
-------------------------------------------------------------------------------
 - Return Inhibition
-------------------------------------------------------------------------------
"""

class DecayKernel:

    def __init__(
        self,
        location: ArrayLike,
        tau_t: float = 5.0,
        tau_s: float = 0.025,
        w_t_min: float = 0.1,
        t: int = 0,
    ):
        self.location = location
        self.tau_t = tau_t
        self.tau_s = tau_s
        self.w_t_min = w_t_min
        self.t = t
        self._expired = False

    @property
    def location(self) -> np.ndarray:
        return self._location

    @location.setter
    def location(self, value: ArrayLike) -> None:
        self._location = np.asarray(value)

    @property
    def tau_t(self) -> float:
        return self._tau_t

    @tau_t.setter
    def tau_t(self, value: float) -> None:
        self._tau_t = value
        self._lam_t = self._tau_t / np.log(2)

    @property
    def tau_s(self) -> float:
        return self._tau_s

    @tau_s.setter
    def tau_s(self, value: float) -> None:
        self._tau_s = value
        self._lam_s = self._tau_s / np.log(2)

    @property
    def expired(self) -> bool:
        return self._expired

    def w_t(self) -> float | np.ndarray:
        """Compute the time-dependent weight at the current step.

        The weight is computed as `exp(-t / lam)`, where `t` is the number of
        steps since the kernel was created, and `lam` is equal to `tau_t / log(2)`.

        Returns:
            The weight, bounded to [0, 1].
        """
        return np.exp(-self.t / self._lam_t)

    def w_s(self, point: np.ndarray) -> float | np.ndarray:
        """Compute the distance-dependent weight.

        The weight is computed as `exp(-z / lam)`, where `z` is the distance
        between the kernel's center and the given point(s), and `lam` is equal
        to `tau_s / log(2)`.

        Args:
            point: One or more 3D vectors. If multiple vectors are provided,
                they must be row vectors (i.e., `point` must be shaped like
                (num_points, 3)).

        Returns:
            The weight(s), bounded to [0, 1]. If `point` is a 1D array, the
            returned weight is a scalar. If `point` is a 2D array, the returned
            weight is a 1D array with shape (num_points,).
        """
        return np.exp(-self._distance(point) / self._lam_s)


    def reset(self) -> None:
        """Reset the kernel to its initial state."""
        self.t = 0
        self._expired = False


    def step(self) -> None:
        """Increment the step counter, and check if the kernel is expired."""
        self.t += 1
        self._expired = self.w_t() < self.w_t_min


    def _distance(self, point: np.ndarray) -> float | np.ndarray:
        """Compute the distance between the kernel's location and one or more points.

        Args:
            point: One or more 3D vectors. If multiple vectors are provided,
                they must be row vectors (i.e., `point` must be shaped like
                (num_points, 3)).

        Returns:
            The distance(s). If `point` is a 1D array, a scalar is returned.
            If `point` is a 2D array, a 1D array with shape (num_points,) is
            returned.
        """
        axis = 1 if point.ndim > 1 else None
        return np.linalg.norm(self._location - point, axis=axis)


    def __call__(self, point: np.ndarray) -> float | np.ndarray:
        """Compute the time- and distance-dependent weight at a given point.

        Computes 1 - w_t * w_s. We subtract the product from 1 because outputs
        are intended to be weights/coefficients. So when w_t and w_s are large,
        the coefficient will drive values towards 0.

        Args:
            point: One or more 3D vectors. If multiple vectors are provided,
                they must be provided as a 2D array with shape (num_points, 3).

        Returns:
            The weight(s) bound to [0, 1]. If `point` is a 1D array, the
            returned weight is a scalar. If `point` is a 2D array, the returned
            weight is a 1D array with shape (num_points,).
        """
        return 1 - (self.w_t() * self.w_s(point))


# class DecayField:
#     """Implements inhibition of return.

#     Used to weight `GoalState.confidence` values.
#     """
#     def __init__(
#         self,
#         kernel_factory: Callable[[Any, ...], DecayKernel] = DecayKernel,
#         kernel_args: dict | None = None,
#         save_telemetry: bool = False,
#         ):
#         self.kernel_factory = kernel_factory
#         self.kernel_args = dict(kernel_args) if kernel_args else {}
#         self.kernels = []
#         self.save_telemetry = save_telemetry
#         self.telemetry = []

#     def reset(self) -> None:
#         self.kernels = []
#         self.telemetry = []

#     def add(self, location: np.ndarray, **kwargs) -> None:
#         """Add a kernel to the field."""
#         if kwargs:
#             kernel_args = {**self.kernel_args, **kwargs}
#         else:
#             kernel_args = self.kernel_args
#         kernel = self.kernel_factory(location, **kernel_args)
#         self.kernels.append(kernel)

#     def step(self) -> None:
#         """Step each kernel, and keep only non-expired ones."""
#         for k in self.kernels:
#             k.step()
#         self.kernels = [k for k in self.kernels if not k.expired]

#     def __call__(self, point: np.ndarray) -> float | np.ndarray:
#         if not self.kernels:
#             return 1.0 if point.ndim == 1 else np.ones(point.shape[0])
#         if len(self.kernels) == 1:
#             return self.kernels[0](point)

#         # Stack kernel parameters and compute in batch
#         results = np.array([k(point) for k in self.kernels])
#         return combine_decay_values(results)

class DecayField:
    """Implements inhibition of return.

    Used to weight `GoalState.confidence` values.
    """
    def __init__(
        self,
        kernel_factory: Callable[[Any, ...], DecayKernel] = DecayKernel,
        kernel_args: dict | None = None,
        save_telemetry: bool = False,
        ):
        self.kernel_factory = kernel_factory
        self.kernel_args = dict(kernel_args) if kernel_args else {}
        self.kernels = []
        self.save_telemetry = save_telemetry
        self.telemetry = []

        self.N = 100
        self.location = np.zeros((self.N, 3))
        self.where = np.zeros(self.N, dtype=bool)
        self.tau_t = np.full(self.N, 5.0)
        self.tau_s = np.full(self.N, 0.025)
        self.w_t_min = np.full(self.N, 0.1)
        self.t = np.zeros(self.N)
        self.expired = np.zeros(self.N, dtype=bool)

        self.w_t = np.zeros(self.N)
        self.w_s = np.zeros(self.N)
        self.w = np.zeros(self.N)

    def reset(self) -> None:
        self.kernels = []
        self.telemetry = []

    def add(self, location: np.ndarray, **kwargs) -> None:
        """Add a kernel to the field."""
        # find first index where 'where' is false
        inds = np.argwhere(~self.where)
        if inds.size == 0:
            # double capacity
            self.N *= 2
            self.location = np.resize(self.location, (self.N, 3))
            self.where = np.resize(self.where, self.N)
            self.tau_t = np.resize(self.tau_t, self.N)
            self.tau_s = np.resize(self.tau_s, self.N)
            self.w_t_min = np.resize(self.w_t_min, self.N)
            self.t = np.resize(self.t, self.N)
            self.expired = np.resize(self.expired, self.N)
            self.w_t = np.resize(self.w_t, self.N)
            self.w_s = np.resize(self.w_s, self.N)
            self.w = np.resize(self.w, self.N)
        else:
            ind = inds[0][0]
            self.location[ind] = location
            self.where[ind] = True
            self.t[ind] = 0
            self.expired[ind] = False
            self.w_t[ind] = 1.0
            self.w_s[ind] = 1.0
            self.w[ind] = 1.0


    def step(self) -> None:
        """Step each kernel, and keep only non-expired ones."""
        self.t += 1
        self.expired = self.w_t < self.w_t_min
        self.where = self.where & ~self.expired

    def __call__(self, point: np.ndarray) -> float | np.ndarray:
        if not self.where.any():
            return 1.0 if point.ndim == 1 else np.ones(point.shape[0])
        lam_t = self.tau_t / np.log(2)
        lam_s = self.tau_s / np.log(2)

        np.exp(-self.t / lam_t, out=self.w_t, where=self.where)

        np.exp(-np.linalg.norm(point - self.location, axis=1) / lam_s,
            out=self.w_s, where=self.where
        )
        np.multiply(self.w_t, self.w_s, out=self.w, where=self.where)
        self.w = 1 - self.w
        w = self.w[self.where]
        return np.min(w, axis=0)


def combine_decay_values(data: np.ndarray) -> np.ndarray:
    return np.min(data, axis=0)
