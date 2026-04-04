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
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from tbp.monty.frameworks.models.grid_cell_matching.cortical_scaffold import (
    CorticalScaffold,
)
from tbp.monty.frameworks.models.grid_cell_matching.grid_modules import (
    GridCellConfig,
    GridModuleArray,
)
from tbp.monty.frameworks.models.grid_cell_matching.rotation_subsystem import (
    RotationSubsystem,
)
from tbp.monty.frameworks.models.grid_cell_matching.sdr_encoder import SDREncoder

__all__ = ["Hypothesis", "HypothesisManager"]

logger = logging.getLogger(__name__)


@dataclass
class Hypothesis:
    """A single hypothesis about object identity and pose.

    Each hypothesis posits: "I am at grid phases Phi on object_id, and the
    object is at rotation R relative to my body frame."

    Attributes:
        object_id: String identifier for the hypothesised object (graph-
            compatible ID for MontyForGraphMatching compatibility).
        rotation: 3x3 rotation matrix R_k. The object's rotation relative
            to the body frame.
        grid_phases: List of 2D phase vectors (one per grid module). These
            are path-integrated per hypothesis using the hypothesis-rotated
            displacement.
        state: Behavioral state index (for future state-conditioned models).
        evidence: Accumulated evidence score. Positive = consistent with
            observations, negative = inconsistent.
        age: Number of matching steps since this hypothesis was created.
        accumulated_displacement: 3D body-frame accumulated displacement,
            used for CMP output (grid phases are internal).
        evidence_history: Recent evidence values for slope computation.
    """

    object_id: str
    rotation: np.ndarray
    grid_phases: list[np.ndarray]
    state: int = 0
    evidence: float = 0.0
    age: int = 0
    accumulated_displacement: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )
    evidence_history: deque = field(
        default_factory=lambda: deque(maxlen=10)
    )

    @property
    def evidence_slope(self) -> float:
        """Compute evidence slope from recent history.

        Uses simple linear regression on the evidence history deque.
        Returns 0 if insufficient history.
        """
        if len(self.evidence_history) < 3:
            return 0.0
        y = np.array(self.evidence_history)
        x = np.arange(len(y), dtype=np.float64)
        x_mean = x.mean()
        y_mean = y.mean()
        denom = np.sum((x - x_mean) ** 2)
        if denom < 1e-10:
            return 0.0
        return float(np.sum((x - x_mean) * (y - y_mean)) / denom)

    def copy(self) -> Hypothesis:
        """Create a deep copy of this hypothesis."""
        return Hypothesis(
            object_id=self.object_id,
            rotation=self.rotation.copy(),
            grid_phases=[p.copy() for p in self.grid_phases],
            state=self.state,
            evidence=self.evidence,
            age=self.age,
            accumulated_displacement=self.accumulated_displacement.copy(),
            evidence_history=deque(self.evidence_history, maxlen=10),
        )


class HypothesisManager:
    """Manages hypothesis lifecycle: initialisation, evidence update, pruning.

    This is the core inference engine. On first observation, it seeds
    hypotheses by querying the cortical scaffold. On subsequent observations,
    it path-integrates each hypothesis with the hypothesis-rotated
    displacement (P0 #1 fix), recalls expected features from the scaffold,
    and computes evidence from morphological alignment (can add/subtract)
    and feature SDR overlap (can only add) (P1 #6 fix).

    Attributes:
        sdr_encoder: Encodes features into SDRs for scaffold comparison.
        rotation_subsystem: Aligns pose vectors for rotation hypotheses.
        grid_config: Grid module configuration (periods, projections).
    """

    def __init__(
        self,
        sdr_encoder: SDREncoder,
        rotation_subsystem: RotationSubsystem,
        grid_config: GridCellConfig,
        past_weight: float = 1.0,
        present_weight: float = 1.0,
    ):
        self.sdr_encoder = sdr_encoder
        self.rotation_subsystem = rotation_subsystem
        self.grid_config = grid_config
        self.past_weight = past_weight
        self.present_weight = present_weight

    def initialise_from_observation(
        self,
        observation: dict,
        cortical_scaffold: CorticalScaffold,
        grid_modules: GridModuleArray,
        known_object_ids: list[str],
    ) -> list[Hypothesis]:
        """Seed initial hypotheses from the first observation.

        Two-stage process:
        1. Query scaffold via recall_from_cue with the sensory SDR.
           If convergence succeeds, recover grid phases and stored features.
        2. For each known object and each compatible rotation (from pose
           vector alignment), create a Hypothesis.

        If the scaffold has no stored content (first training episode),
        returns an empty list — the system should immediately transition
        to exploratory mode.

        Args:
            observation: Dict with keys 'surface_normal', 'curvature_dir',
                'non_morph_features', 'pc1_is_pc2'.
            cortical_scaffold: The scaffold memory to query.
            grid_modules: Grid modules (for projection matrices).
            known_object_ids: List of object IDs currently in memory.

        Returns:
            List of initial hypotheses.
        """
        if not cortical_scaffold.has_stored_content():
            logger.info("Scaffold empty — no hypotheses to initialise")
            return []

        hypotheses = []

        # Encode current observation as SDR
        morph_features = self._get_morph_features(observation)
        non_morph_features = observation.get("non_morph_features")
        sensed_sdr = self.sdr_encoder.encode(morph_features, non_morph_features)

        # Stage: Cortical cue matching
        phases, converged = cortical_scaffold.recall_phases_from_cue(sensed_sdr)

        if converged and phases is not None:
            # Recover stored SDR at recalled phases
            grid_state = self._phases_to_grid_state(phases)
            stored_sdr = cortical_scaffold.recall_at_phase(grid_state)

            # Extract stored pose vectors from the morphological portion
            stored_morph = stored_sdr[:self.sdr_encoder.morph_dim]

            for object_id in known_object_ids:
                # Get stored pose vectors (approximate from SDR — in a full
                # implementation we'd store pose vectors separately, but for
                # now we use the observation's own vectors as a proxy for
                # creating initial rotation hypotheses)
                sensed_normal = observation.get("surface_normal")
                sensed_curv = observation.get("curvature_dir")

                if sensed_normal is None or sensed_curv is None:
                    # No pose information — create hypothesis with identity
                    hypotheses.append(Hypothesis(
                        object_id=object_id,
                        rotation=np.eye(3),
                        grid_phases=[p.copy() for p in phases],
                    ))
                    continue

                pc1_is_pc2 = observation.get("pc1_is_pc2", False)

                # Use sensed pose vectors for both sensed and stored
                # (on first observation, the best we can do is identity-like
                # rotations; as more observations arrive, evidence
                # discriminates)
                rotations = self.rotation_subsystem.initialise_rotations(
                    sensed_normal=sensed_normal,
                    sensed_curvature_dir=sensed_curv,
                    stored_normal=sensed_normal,
                    stored_curvature_dir=sensed_curv,
                    pc1_is_pc2=pc1_is_pc2,
                )

                for R in rotations:
                    hypotheses.append(Hypothesis(
                        object_id=object_id,
                        rotation=R,
                        grid_phases=[p.copy() for p in phases],
                    ))

        if not hypotheses:
            # Fallback: create identity hypotheses for all objects at origin
            origin_phases = [np.zeros(2) for _ in range(self.grid_config.num_modules)]
            for object_id in known_object_ids:
                hypotheses.append(Hypothesis(
                    object_id=object_id,
                    rotation=np.eye(3),
                    grid_phases=[p.copy() for p in origin_phases],
                ))

        logger.info(
            f"Initialised {len(hypotheses)} hypotheses for "
            f"{len(known_object_ids)} objects"
        )
        return hypotheses

    def path_integrate_and_update_evidence(
        self,
        hypotheses: list[Hypothesis],
        displacement: np.ndarray,
        observation: dict,
        cortical_scaffold: CorticalScaffold,
        grid_modules: GridModuleArray,
    ) -> list[Hypothesis]:
        """Path-integrate and update evidence for all hypotheses.

        P0 #1 FIX: Each hypothesis has its displacement rotated by R_k^T
        (inverse of hypothesis rotation) before path integration. This
        transforms the body-frame displacement into the hypothesised
        object-centric frame.

        P1 #6 FIX: Morphology evidence (from pose vector alignment) can
        be positive or negative [-1, 1]. Feature evidence (from SDR
        overlap) can only be positive [0, 1].

        Args:
            hypotheses: Current hypothesis list.
            displacement: Body-frame displacement (3D vector).
            observation: Dict with pose vectors and features.
            cortical_scaffold: Scaffold memory for recall.
            grid_modules: Grid modules (for projection matrices).

        Returns:
            Updated hypothesis list (same objects, modified in-place).
        """
        morph_features = self._get_morph_features(observation)
        non_morph_features = observation.get("non_morph_features")

        # Encode current observation
        sensed_morph_sdr = self.sdr_encoder.encode_morphological(morph_features)
        sensed_feat_sdr = self.sdr_encoder.encode_non_morphological(non_morph_features)

        sensed_normal = observation.get("surface_normal")
        sensed_curv = observation.get("curvature_dir")

        morph_dim = self.sdr_encoder.morph_dim

        for hyp in hypotheses:
            # ---- P0 #1: Rotate displacement by hypothesis rotation ----
            # R_k^{-1} = R_k^T for rotation matrices
            rotation_inv = hyp.rotation.T
            rotated_displacement = rotation_inv @ displacement

            # Path-integrate hypothesis grid phases with rotated displacement
            for m in range(self.grid_config.num_modules):
                phase_shift = grid_modules.projections[m] @ rotated_displacement
                hyp.grid_phases[m] = (
                    (hyp.grid_phases[m] + phase_shift)
                    % self.grid_config.periods[m]
                )

            # Update accumulated displacement (body frame, for CMP output)
            hyp.accumulated_displacement += displacement

            # Recall stored SDR at updated grid phase
            grid_state = self._phases_to_grid_state(hyp.grid_phases)
            stored_sdr = cortical_scaffold.recall_at_phase(grid_state)

            # ---- Compute evidence ----
            evidence_update = 0.0

            if np.sum(np.abs(stored_sdr)) < 1e-8:
                # Nothing stored at this phase — strong negative evidence
                evidence_update = -1.0
            else:
                # ---- P1 #6: Separate morph and non-morph evidence ----

                # Morphology evidence: can be positive or negative [-1, 1]
                if sensed_normal is not None and sensed_curv is not None:
                    stored_morph_sdr = stored_sdr[:morph_dim]
                    morph_overlap = SDREncoder.sdr_overlap(
                        sensed_morph_sdr, stored_morph_sdr
                    )
                    # Also compute pose vector alignment evidence
                    morph_ev = RotationSubsystem.morphology_evidence(
                        sensed_normal, sensed_curv,
                        sensed_normal, sensed_curv,  # Stored pose vectors
                        hyp.rotation,
                    )
                    # Combine SDR overlap and pose alignment
                    evidence_update += morph_ev
                else:
                    # No pose vectors — use SDR overlap only
                    morph_overlap = SDREncoder.sdr_overlap(
                        sensed_morph_sdr, stored_sdr[:morph_dim]
                    )
                    evidence_update += 2.0 * morph_overlap - 1.0

                # Feature evidence: can only be positive [0, 1]
                stored_feat_sdr = stored_sdr[morph_dim:]
                feat_ev = SDREncoder.sdr_overlap(sensed_feat_sdr, stored_feat_sdr)
                evidence_update += feat_ev

            # Update evidence with past/present weighting
            hyp.evidence = (
                self.past_weight * hyp.evidence
                + self.present_weight * evidence_update
            )

            hyp.age += 1
            hyp.evidence_history.append(hyp.evidence)

        return hypotheses

    def detect_novelty(
        self,
        cortical_scaffold: CorticalScaffold,
        sensory_sdr: np.ndarray,
    ) -> bool:
        """Detect novelty via scaffold convergence failure.

        If the scaffold cannot converge when cued with the current
        observation, the sensory input is genuinely novel.

        Args:
            cortical_scaffold: Scaffold memory to test.
            sensory_sdr: Current sensory SDR.

        Returns:
            True if the observation is novel (scaffold failed to converge).
        """
        _, converged = cortical_scaffold.recall_from_cue(sensory_sdr)
        return not converged

    def _get_morph_features(self, observation: dict) -> np.ndarray | None:
        """Extract morphological features from observation dict.

        Concatenates surface normal (3) and curvature direction (3) into
        a single 6D vector.
        """
        normal = observation.get("surface_normal")
        curv_dir = observation.get("curvature_dir")
        if normal is None or curv_dir is None:
            return None
        return np.concatenate([normal, curv_dir])

    def _phases_to_grid_state(
        self, phases: list[np.ndarray]
    ) -> np.ndarray:
        """Convert list of 2D phases to binary grid state vector.

        Args:
            phases: List of 2D phase vectors, one per module.

        Returns:
            Binary grid state of shape (grid_dim,).
        """
        state = np.zeros(self.grid_config.grid_dim, dtype=np.float64)
        offset = 0
        for m, period in enumerate(self.grid_config.periods):
            ix = int(np.floor(phases[m][0])) % period
            iy = int(np.floor(phases[m][1])) % period
            state[offset + ix * period + iy] = 1.0
            offset += period**2
        return state
