# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

__all__ = ["GridCellConfig", "GridModuleArray"]


@dataclass
class GridCellConfig:
    """Configuration for grid cell modules.

    Attributes:
        num_modules: Number of grid modules. Each has its own period and random
            projection matrix.
        periods: Toroidal periods for each module. Should be coprime for
            exponential capacity (product of period^2 unique states).
        projection_dim: Dimensionality of input displacement vectors. 3 for
            physical 3D space, but can be higher for abstract spaces.
        seed: Random seed for reproducible projection matrices.
    """

    num_modules: int = 6
    periods: list[int] = field(
        default_factory=lambda: [7, 8, 9, 11, 13, 16]
    )
    projection_dim: int = 3
    seed: int = 42

    def __post_init__(self):
        if len(self.periods) != self.num_modules:
            raise ValueError(
                f"Number of periods ({len(self.periods)}) must match "
                f"num_modules ({self.num_modules})"
            )

    @property
    def grid_dim(self) -> int:
        """Total dimensionality of the binary grid state vector."""
        return sum(p**2 for p in self.periods)


class GridModuleArray:
    """Array of M grid modules with coprime toroidal periods.

    Handles TRANSLATION ONLY via additive path integration on toroidal
    lattices. Each module maintains a 2D continuous phase on a lambda_m x
    lambda_m torus. Path integration projects a d-dimensional displacement
    into 2D via a fixed random projection matrix, then shifts the phase
    modularly.

    Rotation is handled separately by the RotationSubsystem — grid cells
    encode translation, head direction cells encode orientation. This matches
    the biological separation.

    Based on Klukas et al. (2020): random projections from R^d to R^2
    preserve neighbourhood structure with high probability for any d.
    """

    def __init__(self, config: GridCellConfig):
        self.config = config
        rng = np.random.default_rng(config.seed)

        # Fixed random projections: d-dimensional -> 2D per module
        self.projections = [
            rng.standard_normal((2, config.projection_dim))
            for _ in range(config.num_modules)
        ]

        # Current phases: 2D continuous coordinates per module
        self.phases = [np.zeros(2) for _ in range(config.num_modules)]

    def path_integrate(
        self,
        displacement: np.ndarray,
        rotation: np.ndarray | None = None,
    ) -> None:
        """Update phases given a displacement vector.

        The displacement is first rotated by the hypothesis rotation (if
        provided), then projected into 2D per module. This implements:

            phi_m(t+1) = (phi_m(t) + A_m @ R_k^{-1} @ dx) mod lambda_m

        For body-frame tracking (no hypothesis), pass rotation=None.
        For hypothesis-specific tracking, pass rotation=R_k.T (inverse
        of the hypothesis rotation, which equals the transpose for
        rotation matrices).

        Args:
            displacement: d-dimensional displacement vector (body frame).
            rotation: Optional rotation matrix to apply before projection.
                Should be R_k^{-1} = R_k^T for hypothesis k. If None,
                displacement is used as-is (body frame).
        """
        if rotation is not None:
            rotated = rotation @ displacement
        else:
            rotated = displacement

        for m in range(self.config.num_modules):
            phase_shift = self.projections[m] @ rotated
            self.phases[m] = (
                (self.phases[m] + phase_shift) % self.config.periods[m]
            )

    def get_binary_state(self) -> np.ndarray:
        """Convert continuous phases to M-hot binary vector for scaffold input.

        Each module contributes one active bit at the discretised lattice
        position. The total vector length is sum(period^2 for each module).

        Returns:
            Binary vector of shape (grid_dim,) with exactly num_modules
            active bits.
        """
        state = np.zeros(self.config.grid_dim, dtype=np.float64)
        offset = 0
        for m, period in enumerate(self.config.periods):
            ix = int(np.floor(self.phases[m][0])) % period
            iy = int(np.floor(self.phases[m][1])) % period
            state[offset + ix * period + iy] = 1.0
            offset += period**2
        return state

    def set_phases(self, phases: list[np.ndarray]) -> None:
        """Set phases directly for hypothesis evaluation.

        Args:
            phases: List of 2D phase vectors, one per module.
        """
        self.phases = [p.copy() for p in phases]

    def get_phases(self) -> list[np.ndarray]:
        """Get a copy of current phases.

        Returns:
            List of 2D phase vectors, one per module.
        """
        return [p.copy() for p in self.phases]

    def reset(self) -> None:
        """Reset all phases to zero (origin)."""
        self.phases = [np.zeros(2) for _ in range(self.config.num_modules)]

    def copy(self) -> GridModuleArray:
        """Create a deep copy for temporary phase tracking.

        Returns:
            Independent copy with same config and projections but
            independent phase state.
        """
        return copy.deepcopy(self)

    @staticmethod
    def toroidal_distance(
        phi_a: list[np.ndarray],
        phi_b: list[np.ndarray],
        periods: list[int],
    ) -> float:
        """Compute distance between two phase vectors on the product torus.

        For each module, the distance wraps around the torus edges. The
        total distance is the Euclidean norm of per-module wrapped
        differences.

        Args:
            phi_a: First set of phases (one 2D vector per module).
            phi_b: Second set of phases (one 2D vector per module).
            periods: Toroidal period for each module.

        Returns:
            Non-negative scalar distance.
        """
        total = 0.0
        for m, period in enumerate(periods):
            diff = np.abs(phi_a[m] - phi_b[m])
            wrapped = np.minimum(diff, period - diff)
            total += np.sum(wrapped**2)
        return np.sqrt(total)

    @staticmethod
    def shift_phases(
        phases: list[np.ndarray],
        displacement: np.ndarray,
        projections: list[np.ndarray],
        periods: list[int],
    ) -> list[np.ndarray]:
        """Shift phases by a projected displacement (for vote transformation).

        This is used during voting: the inter-sensor displacement is
        projected into phase space and added modularly to the vote's phases.

        Args:
            phases: Input phases to shift.
            displacement: d-dimensional displacement to project and add.
            projections: Random projection matrices (one per module).
            periods: Toroidal periods.

        Returns:
            New shifted phases.
        """
        shifted = []
        for m, period in enumerate(periods):
            phase_shift = projections[m] @ displacement
            new_phase = (phases[m] + phase_shift) % period
            shifted.append(new_phase)
        return shifted
