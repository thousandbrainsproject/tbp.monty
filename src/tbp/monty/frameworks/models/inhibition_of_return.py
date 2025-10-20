# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import numpy as np


class DecayKernel:
    """Decay kernel represents a previously visited location.

    Returns the product of an time- and space- dependent exponentials.
    """

    def __init__(
        self,
        location: np.ndarray,
        tau_t: float = 10.0,
        tau_s: float = 0.01,
        spatial_cutoff: float | None = 0.02,
        w_t_min: float = 0.1,
    ):
        self._location = location
        self._tau_t = tau_t
        self._tau_s = tau_s
        self._spatial_cutoff = spatial_cutoff
        self._w_t_min = w_t_min
        self._t = 0

    def w_t(self) -> float:
        """Compute the time-dependent weight at the current step.

        The weight is computed as `exp(-t / lam)`, where `t` is the number of
        steps since the kernel was created, and `lam` is equal to `tau_t / log(2)`.

        Returns:
            The weight, bounded to [0, 1].
        """
        return np.exp(-self._t / (self._tau_t / float(np.log(2))))

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
            dist = self.distance(point)
            if self._spatial_cutoff is not None and dist > self._spatial_cutoff:
                return 0.0
            return np.exp(-dist / (self._tau_s / np.log(2)))
        else:
            dist = self.distance(point)
            out = np.exp(-dist / (self._tau_s / np.log(2)))
            if self._spatial_cutoff is not None:
                out[dist > self._spatial_cutoff] = 0.0
            return out

    def step(self) -> bool:
        """Increment the step counter, and check if the kernel is expired.

        Returns:
            True if the kernel is expired, False otherwise.
        """
        self._t += 1
        return self.w_t() < self._w_t_min

    def distance(self, point: np.ndarray) -> float | np.ndarray:
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

        Computes the product of the time- and distance-dependent weights. Weights
        are bounded to [0, 1], where values close to 1 indicate the kernel has a
        large influence on the given point(s).

        Args:
            point: One or more 3D vectors. If multiple vectors are provided,
                they must be provided as a 2D array with shape (num_points, 3).

        Returns:
            The weight(s) bound to [0, 1]. If `point` is a 1D array, the
            returned weight is a scalar. If `point` is a 2D array, the returned
            weight is a 1D array with shape (num_points,).
        """
        return self.w_t() * self.w_s(point)


class DecayKernelFactory:
    def __init__(
        self,
        tau_t: float = 10.0,
        tau_s: float = 0.01,
        spatial_cutoff: float | None = 0.02,
        w_t_min: float = 0.1,
    ):
        self._tau_t = tau_t
        self._tau_s = tau_s
        self._spatial_cutoff = spatial_cutoff
        self._w_t_min = w_t_min

    def __call__(self, location: np.ndarray) -> DecayKernel:
        return DecayKernel(
            location, self._tau_t, self._tau_s, self._spatial_cutoff, self._w_t_min
        )


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

    def __init__(self, kernel_factory: DecayKernelFactory | None = None):
        if kernel_factory is None:
            kernel_factory = DecayKernelFactory()
        self._kernel_factory = kernel_factory
        self._kernels = []

    def reset(self) -> None:
        self._kernels = []

    def add(self, location: np.ndarray) -> None:
        """Add a kernel to the field."""
        self._kernels.append(self._kernel_factory(location))

    def step(self) -> None:
        """Step each kernel to increment its counter, and keep only non-expired ones."""
        self._kernels = [k for k in self._kernels if not k.step()]

    def compute_weight(self, point: np.ndarray) -> float | np.ndarray:
        if not self._kernels:
            return 0.0 if point.ndim == 1 else np.zeros(point.shape[0])
        if len(self._kernels) == 1:
            return self._kernels[0](point)

        # Stack kernel parameters and compute in batch
        results = np.array([k(point) for k in self._kernels])
        return np.max(results, axis=0)
