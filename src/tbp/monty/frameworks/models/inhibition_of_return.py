# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np
from numpy.typing import ArrayLike

from tbp.monty.frameworks.models.states import GoalState


class DecayKernel:
    """Decay kernel represents a previously visited location.

    Returns the product of an time- and space- dependent exponentials.
    """

    def __init__(
        self,
        location: ArrayLike,
        tau_t: float = 10.0,
        tau_s: float = 0.01,
        cutoff_s: float | None = 0.02,
        w_t_min: float = 0.1,
        t: int = 0,
    ):
        self.location = location
        self.tau_t = tau_t
        self.tau_s = tau_s
        self.cutoff_s = cutoff_s
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
        if point.ndim == 1:
            dist = self._distance(point)
            if self.cutoff_s is not None and dist > self.cutoff_s:
                return 0.0
            return np.exp(-dist / self._lam_s)
        else:
            dist = self._distance(point)
            out = np.exp(-dist / self._lam_s)
            if self.cutoff_s is not None:
                out[dist > self.cutoff_s] = 0.0
            return out

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
        return self.w_t() * self.w_s(point)


class DecayField:
    """Implements inhibition of return.

    Manages a collection of decay kernels. Used to weight
    `GoalState.confidence` values.

    Calling order:
      - add (usually)
      - __call__ (usually many times)
      - update_telemetry
      - step
    """

    def __init__(
        self,
        kernel_factory: Callable[[Any, ...], DecayKernel] = DecayKernel,
        kernel_args: dict | None = None,
    ):
        self.kernel_factory = kernel_factory
        self.kernel_args = dict(kernel_args) if kernel_args else {}
        self.kernels = []

    def reset(self) -> None:
        self.kernels = []

    def add(self, location: np.ndarray, **kwargs) -> None:
        """Add a kernel to the field."""
        if kwargs:
            kernel_args = {**self.kernel_args, **kwargs}
        else:
            kernel_args = self.kernel_args
        kernel = self.kernel_factory(location, **kernel_args)
        self.kernels.append(kernel)

    def step(self) -> None:
        """Step each kernel, and keep only non-expired ones."""
        for k in self.kernels:
            k.step()
        self.kernels = [k for k in self.kernels if not k.expired]

    def __call__(self, point: np.ndarray) -> float | np.ndarray:
        if not self.kernels:
            return 1.0 if point.ndim == 1 else np.ones(point.shape[0])
        if len(self.kernels) == 1:
            return self.kernels[0](point)

        # Stack kernel parameters and compute in batch
        results = np.array([k(point) for k in self.kernels])
        return combine_decay_values(results)


def combine_decay_values(data: np.ndarray) -> np.ndarray:
    return np.max(data, axis=0)


def normalize_confidence(goal_states: Iterable[GoalState]) -> None:
    """Normalize the confidence of the goal states."""
    confidence_values = [goal_state.confidence for goal_state in goal_states]
    max_confidence = max(confidence_values)
    min_confidence = min(confidence_values)
    for goal_state in goal_states:
        goal_state.confidence = (goal_state.confidence - min_confidence) / (
            max_confidence - min_confidence
        )
