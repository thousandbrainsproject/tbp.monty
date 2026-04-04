# Copyright 2025-2026 Thousand Brains Project
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

import numpy as np

from tbp.monty.frameworks.models.grid_cell_matching.grid_modules import GridCellConfig

__all__ = ["CorticalScaffold", "ScaffoldConfig"]

logger = logging.getLogger(__name__)


@dataclass
class ScaffoldConfig:
    """Configuration for the cortical scaffold memory.

    Attributes:
        num_place_cells: Number of place cells in the scaffold. Controls the
            bottleneck between grid and sensory representations.
        sensory_dim: Dimensionality of the sensory SDR layer.
        learning_rate: Damped pseudoinverse learning rate (eta). Values < 1
            average over multiple exposures, preventing single noisy
            observations from dominating.
        convergence_steps: Maximum iterations for recall_from_cue convergence.
        threshold: Place cell activation threshold (theta). Controls sparsity
            of place cell representations.
    """

    num_place_cells: int = 800
    sensory_dim: int = 2048
    learning_rate: float = 0.5
    convergence_steps: int = 10
    threshold: float = 0.1


class CorticalScaffold:
    """Per-LM scaffold memory using the Vector-HaSH architecture.

    All objects are stored in shared weight matrices. Object identity is
    encoded in the sensory SDR bound to each scaffold state. Different
    objects naturally separate in phase space because hypothesis rotations
    produce different phase trajectories from the same physical displacements.

    Weight matrices:
        W_gp (N_p x N_g): Fixed random. Grid -> Place. Converts sparse
            grid state to dense place cell activation.
        W_pg (N_g x N_p): Hebbian then fixed. Place -> Grid. Used for
            WTA-per-module in recall_from_cue convergence loop.
        W_ps (N_s x N_p): Learned via damped pseudoinverse. Place -> Sensory.
            Stores sensory patterns at scaffold addresses.
        W_sp (N_p x N_s): Learned via damped pseudoinverse. Sensory -> Place.
            Used for recall_from_cue (sensory cue -> place activation).

    The critical operation distinction:
        - recall_at_phase: SINGLE matrix multiply, O(N_p * N_s). Used during
          evidence evaluation when hypothesis already specifies the phase.
        - recall_from_cue: ITERATIVE convergence. Used during initialisation
          when phase is unknown and must be recovered from a sensory cue.
    """

    def __init__(
        self,
        grid_config: GridCellConfig,
        scaffold_config: ScaffoldConfig | None = None,
        seed: int = 42,
    ):
        if scaffold_config is None:
            scaffold_config = ScaffoldConfig()

        self.config = scaffold_config
        self.grid_config = grid_config

        N_g = grid_config.grid_dim
        N_p = scaffold_config.num_place_cells
        N_s = scaffold_config.sensory_dim

        self.learning_rate = scaffold_config.learning_rate
        self.threshold = scaffold_config.threshold
        self.convergence_steps = scaffold_config.convergence_steps

        rng = np.random.default_rng(seed)

        # W_gp: grid -> place (fixed random, developmental)
        # Scale by 1/sqrt(N_g) so place cell activations are O(1)
        self.W_gp = rng.standard_normal((N_p, N_g)) * (1.0 / np.sqrt(N_g))

        # W_pg: place -> grid (initialised via Hebbian learning on random
        # grid-place co-activations, then fixed)
        self._init_W_pg(rng, N_g, N_p)

        # W_ps: place -> sensory (learned via damped pseudoinverse)
        self.W_ps = np.zeros((N_s, N_p))

        # W_sp: sensory -> place (learned via damped pseudoinverse)
        self.W_sp = np.zeros((N_p, N_s))

        # Track whether anything has been stored
        self._has_content = False

        # Store module boundary offsets for WTA
        self._module_offsets = []
        offset = 0
        for period in grid_config.periods:
            self._module_offsets.append((offset, period))
            offset += period**2

    def _init_W_pg(self, rng, N_g: int, N_p: int) -> None:
        """Initialise place->grid weights via Hebbian co-activation.

        Simulates a developmental phase where random grid states activate
        place cells (via W_gp), and the co-activation is used to form
        W_pg via Hebbian learning.
        """
        num_developmental_steps = N_p * 2
        W_pg = np.zeros((N_g, N_p))

        for _ in range(num_developmental_steps):
            # Random grid state (one-hot per module)
            g = np.zeros(N_g)
            offset = 0
            for period in self.grid_config.periods:
                idx = rng.integers(0, period**2)
                g[offset + idx] = 1.0
                offset += period**2

            # Place cell activation
            p = np.maximum(self.W_gp @ g - self.threshold, 0)
            if np.sum(p) > 1e-8:
                # Hebbian: W_pg += g * p^T (normalised)
                W_pg += np.outer(g, p) / np.sum(p)

        self.W_pg = W_pg

    def place_from_grid(self, g: np.ndarray) -> np.ndarray:
        """Compute place cell activation from grid state.

        Args:
            g: Binary grid state vector of shape (N_g,).

        Returns:
            Place cell activation vector of shape (N_p,), non-negative.
        """
        return np.maximum(self.W_gp @ g - self.threshold, 0)

    def recall_at_phase(self, grid_state: np.ndarray) -> np.ndarray:
        """Recall stored SDR at a given grid state.

        This is a SINGLE matrix multiply. Used during evidence evaluation
        when the hypothesis already specifies the phase. O(N_p * N_s),
        independent of model complexity.

        Args:
            grid_state: Binary grid state vector of shape (N_g,).

        Returns:
            Recalled sensory pattern of shape (N_s,). May be all zeros
            if nothing is stored at this phase.
        """
        p = self.place_from_grid(grid_state)
        if np.sum(p) < 1e-8:
            return np.zeros(self.W_ps.shape[0])
        return self.W_ps @ p

    def recall_from_cue(
        self, sensory_cue: np.ndarray
    ) -> tuple[np.ndarray, bool]:
        """Recall from partial sensory cue via iterative scaffold convergence.

        Used during INITIALISATION only, when the phase is unknown.
        The sensory cue activates place cells via W_sp, which activate
        grid cells via W_pg (with WTA per module), which activate place
        cells via W_gp, and so on until convergence.

        Args:
            sensory_cue: Sensory SDR of shape (N_s,).

        Returns:
            Tuple of (recalled_sdr, converged):
                - recalled_sdr: Recalled sensory pattern of shape (N_s,).
                - converged: Whether the scaffold converged to a stable
                  fixed point.
        """
        # Sensory -> Place
        p = self.W_sp @ sensory_cue

        prev_g = None
        for _ in range(self.convergence_steps):
            g = self._grid_from_place(p)
            p = self.place_from_grid(g)

            if prev_g is not None and np.array_equal(g, prev_g):
                # Converged to fixed point
                return self.W_ps @ p, True
            prev_g = g.copy()

        # Check if close to convergence (place cells are active)
        if np.sum(p) > self.threshold * 10:
            return self.W_ps @ p, True

        return np.zeros(self.W_ps.shape[0]), False

    def recall_phases_from_cue(
        self, sensory_cue: np.ndarray
    ) -> tuple[list[np.ndarray] | None, bool]:
        """Recall grid phases from a sensory cue.

        Like recall_from_cue, but returns the recovered grid phases instead
        of the sensory recall. Used during hypothesis initialisation to seed
        phase hypotheses.

        Args:
            sensory_cue: Sensory SDR of shape (N_s,).

        Returns:
            Tuple of (phases, converged):
                - phases: List of 2D phase vectors per module, or None if
                  convergence failed.
                - converged: Whether convergence succeeded.
        """
        p = self.W_sp @ sensory_cue

        prev_g = None
        for _ in range(self.convergence_steps):
            g = self._grid_from_place(p)
            p = self.place_from_grid(g)

            if prev_g is not None and np.array_equal(g, prev_g):
                return self._extract_phases_from_grid(g), True
            prev_g = g.copy()

        if np.sum(p) > self.threshold * 10:
            g = self._grid_from_place(p)
            return self._extract_phases_from_grid(g), True

        return None, False

    def store(
        self,
        grid_state: np.ndarray,
        sensory_sdr: np.ndarray,
    ) -> None:
        """Bind sensory SDR to scaffold state via damped pseudoinverse update.

        Updates both W_ps (place->sensory) and W_sp (sensory->place) at the
        configured learning rate eta. The damped update prevents catastrophic
        interference: eta < 1 means new observations are averaged with
        existing patterns.

        Args:
            grid_state: Binary grid state vector of shape (N_g,).
            sensory_sdr: Sensory SDR to store, shape (N_s,).
        """
        p = self.place_from_grid(grid_state)
        pp = float(p @ p)
        if pp < 1e-8:
            return

        # W_ps update: place -> sensory
        error_ps = sensory_sdr - self.W_ps @ p
        self.W_ps += self.learning_rate * np.outer(error_ps, p) / pp

        # W_sp update: sensory -> place
        ss = float(sensory_sdr @ sensory_sdr)
        if ss > 1e-8:
            error_sp = p - self.W_sp @ sensory_sdr
            self.W_sp += self.learning_rate * np.outer(error_sp, sensory_sdr) / ss

        self._has_content = True

    def has_stored_content(self) -> bool:
        """Check if the scaffold has any stored patterns.

        Returns:
            True if store() has been called at least once.
        """
        return self._has_content

    def _grid_from_place(self, p: np.ndarray) -> np.ndarray:
        """Winner-take-all per module from place cell activations.

        Projects place cell activations back to grid space via W_pg, then
        applies WTA within each module (only the maximum activation within
        each module's subspace is kept).

        Args:
            p: Place cell activation vector of shape (N_p,).

        Returns:
            Binary grid state vector of shape (N_g,) with exactly one
            active bit per module.
        """
        raw = self.W_pg @ p
        g = np.zeros_like(raw)

        for offset, period in self._module_offsets:
            module_slice = raw[offset:offset + period**2]
            if np.max(module_slice) > 0:
                winner = np.argmax(module_slice)
                g[offset + winner] = 1.0

        return g

    def _extract_phases_from_grid(
        self, g: np.ndarray
    ) -> list[np.ndarray]:
        """Extract continuous 2D phases from a binary grid state.

        Finds the active bit in each module and converts the linear index
        back to 2D phase coordinates.

        Args:
            g: Binary grid state vector of shape (N_g,).

        Returns:
            List of 2D phase vectors, one per module.
        """
        phases = []
        for offset, period in self._module_offsets:
            module_slice = g[offset:offset + period**2]
            idx = np.argmax(module_slice)
            ix = idx // period
            iy = idx % period
            phases.append(np.array([float(ix), float(iy)]))
        return phases

    def state_dict(self) -> dict:
        """Serialise scaffold state for saving.

        Returns:
            Dictionary containing all weight matrices and configuration.
        """
        return {
            "W_gp": self.W_gp.copy(),
            "W_pg": self.W_pg.copy(),
            "W_ps": self.W_ps.copy(),
            "W_sp": self.W_sp.copy(),
            "has_content": self._has_content,
            "grid_config": {
                "num_modules": self.grid_config.num_modules,
                "periods": self.grid_config.periods,
                "projection_dim": self.grid_config.projection_dim,
                "seed": self.grid_config.seed,
            },
            "scaffold_config": {
                "num_place_cells": self.config.num_place_cells,
                "sensory_dim": self.config.sensory_dim,
                "learning_rate": self.config.learning_rate,
                "convergence_steps": self.config.convergence_steps,
                "threshold": self.config.threshold,
            },
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore scaffold state from a saved dict.

        Args:
            state_dict: Dictionary from a previous state_dict() call.
        """
        self.W_gp = state_dict["W_gp"]
        self.W_pg = state_dict["W_pg"]
        self.W_ps = state_dict["W_ps"]
        self.W_sp = state_dict["W_sp"]
        self._has_content = state_dict["has_content"]
