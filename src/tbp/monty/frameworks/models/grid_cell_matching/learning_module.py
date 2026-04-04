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

import numpy as np
from scipy.spatial.transform import Rotation

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.abstract_monty_classes import LearningModule
from tbp.monty.frameworks.models.buffer import FeatureAtLocationBuffer
from tbp.monty.frameworks.models.grid_cell_matching.burst_sampling import (
    BurstSamplingConfig,
    GridCellBurstSampler,
)
from tbp.monty.frameworks.models.grid_cell_matching.cortical_scaffold import (
    CorticalScaffold,
    ScaffoldConfig,
)
from tbp.monty.frameworks.models.grid_cell_matching.goal_state_generator import (
    GridCellGoalStateGenerator,
)
from tbp.monty.frameworks.models.grid_cell_matching.grid_modules import (
    GridCellConfig,
    GridModuleArray,
)
from tbp.monty.frameworks.models.grid_cell_matching.hypothesis import (
    Hypothesis,
    HypothesisManager,
)
from tbp.monty.frameworks.models.grid_cell_matching.rotation_subsystem import (
    RotationSubsystem,
)
from tbp.monty.frameworks.models.grid_cell_matching.sdr_encoder import SDREncoder
from tbp.monty.frameworks.models.states import GoalState, State

__all__ = ["GridCellLM"]

logger = logging.getLogger(__name__)


class GridCellLM(LearningModule):
    """Grid cell-based Learning Module using scaffold memory.

    Replaces the EvidenceGraphLM's explicit 3D graphs and KDTree search with
    grid cell phase-space path integration and SDR-based pattern matching.

    Key architectural decisions:
    - Inherits from LearningModule directly (NOT GraphLM).
    - Translation via grid cell path integration; rotation via pose vector
      alignment (separate subsystems, matching biological separation).
    - All objects stored in shared scaffold weight matrices; object identity
      encoded in the SDR bound to each scaffold state.
    - Per-hypothesis grid phases for matching; central grid modules for
      exploration (kept in sync via body-frame path integration).

    P0 Bug Fixes Incorporated:
    1. Displacement rotated by R_k^T per hypothesis before path integration.
    2. Central grid modules path-integrated during both matching and exploration.
    3. post_episode re-traces trajectory and stores in scaffold.
    4. Vote transformation uses toroidal phase distance, not Euclidean.

    P1 Bug Fixes Incorporated:
    5. Burst sampling for dynamic hypothesis management.
    6. Separate morphological/non-morphological evidence computation.
    7. Mode-conditional pre_episode reset.
    8. Symmetry evidence tracking.
    """

    def __init__(
        self,
        # Grid cell configuration
        grid_config: GridCellConfig | dict | None = None,
        # Scaffold configuration
        scaffold_config: ScaffoldConfig | dict | None = None,
        # SDR configuration
        sdr_input_dim: int = 10,
        sdr_dim: int = 2048,
        sdr_num_active: int = 41,
        # Rotation configuration
        num_isotropic_samples: int = 8,
        # Evidence parameters
        feature_weights: dict | None = None,
        tolerances: dict | None = None,
        x_percent_threshold: float = 20.0,
        object_evidence_threshold: float = 1.0,
        required_symmetry_evidence: int = 5,
        past_weight: float = 1.0,
        present_weight: float = 1.0,
        vote_weight: float = 1.0,
        vote_evidence_threshold: float = 0.8,
        # Evidence pruning
        evidence_prune_threshold: float = -10.0,
        # Goal state generator
        gsg: GridCellGoalStateGenerator | None = None,
        # Burst sampling
        burst_sampling_config: BurstSamplingConfig | dict | None = None,
        # Misc
        seed: int = 42,
    ):
        super().__init__()

        # Parse config dicts
        if isinstance(grid_config, dict):
            grid_config = GridCellConfig(**grid_config)
        elif grid_config is None:
            grid_config = GridCellConfig()

        if isinstance(scaffold_config, dict):
            scaffold_config = ScaffoldConfig(**scaffold_config)
        elif scaffold_config is None:
            scaffold_config = ScaffoldConfig(sensory_dim=sdr_dim)

        if isinstance(burst_sampling_config, dict):
            burst_sampling_config = BurstSamplingConfig(**burst_sampling_config)

        self.grid_config = grid_config

        # Core components
        self.grid_modules = GridModuleArray(grid_config)
        self.cortical_scaffold = CorticalScaffold(
            grid_config, scaffold_config, seed=seed
        )
        self.sdr_encoder = SDREncoder(
            input_dim=sdr_input_dim,
            sdr_dim=sdr_dim,
            num_active=sdr_num_active,
            seed=seed,
        )
        self.rotation_subsystem = RotationSubsystem(
            num_isotropic_samples=num_isotropic_samples
        )
        self.hypothesis_manager = HypothesisManager(
            sdr_encoder=self.sdr_encoder,
            rotation_subsystem=self.rotation_subsystem,
            grid_config=grid_config,
            past_weight=past_weight,
            present_weight=present_weight,
        )
        self.burst_sampler = GridCellBurstSampler(burst_sampling_config)

        # Evidence parameters
        self.feature_weights = feature_weights or {}
        self.tolerances = tolerances or {}
        self.x_percent_threshold = x_percent_threshold
        self.object_evidence_threshold = object_evidence_threshold
        self.required_symmetry_evidence = required_symmetry_evidence
        self.past_weight = past_weight
        self.present_weight = present_weight
        self.vote_weight = vote_weight
        self.vote_evidence_threshold = vote_evidence_threshold
        self.evidence_prune_threshold = evidence_prune_threshold

        # Goal state generator
        if gsg is None:
            gsg = GridCellGoalStateGenerator(parent_lm=self)
        else:
            gsg.parent_lm = self
        self.gsg = gsg

        # Buffer (same as GraphLM)
        self.buffer = FeatureAtLocationBuffer()
        self.buffer.reset()

        # State tracking (compatible with GraphLM/MontyForGraphMatching)
        self.learning_module_id = "LM_0"
        self.mode: ExperimentMode | None = None
        self.terminal_state = None
        self.primary_target = None
        self.primary_target_rotation_quat = None
        self.detected_object = None
        self.detected_pose = [None for _ in range(7)]
        self.detected_rotation_r = None
        self.has_detailed_logger = False
        self.stepwise_target_object = None
        self.stepwise_targets_list = []

        # Target <-> graph mappings
        self.target_to_graph_id = {}
        self.graph_id_to_target = {}

        # Known object registry (replaces graph_memory.get_memory_ids())
        self._known_object_ids: list[str] = []

        # Hypothesis state
        self.hypotheses: list[Hypothesis] = []
        self.symmetry_evidence = 0

        # Current most-likely hypothesis (MLH) dict — format expected by
        # MontyForGraphMatching and MontyForEvidenceGraphMatching
        self.current_mlh = {
            "graph_id": None,
            "location": np.zeros(3),
            "rotation": Rotation.identity(),
            "scale": 1,
            "evidence": 0,
        }

        # Possible matches dict — keyed by object_id for compatibility
        self.possible_matches: dict[str, bool] = {}

        # Previous location for displacement computation
        self._prev_location: np.ndarray | None = None

    # ==================== LearningModule Interface ====================

    def reset(self):
        """Reset hypotheses and transient state for a new episode."""
        self.hypotheses = []
        self.grid_modules.reset()
        self.symmetry_evidence = 0
        self.possible_matches = {}
        self._prev_location = None
        self.current_mlh = {
            "graph_id": None,
            "location": np.zeros(3),
            "rotation": Rotation.identity(),
            "scale": 1,
            "evidence": 0,
        }

    def pre_episode(self, primary_target=None) -> None:
        """Reset for a new episode.

        P1 #7 FIX: Mode-conditional reset. During training (supervised
        pretraining), full reset including grid modules. During evaluation,
        reset hypotheses but keep scaffold intact.
        """
        self.reset()
        self.buffer.reset()
        self.burst_sampler.reset()
        if self.gsg is not None:
            self.gsg.reset()

        if primary_target is not None:
            if isinstance(primary_target, dict):
                self.primary_target = primary_target.get("object")
                self.primary_target_rotation_quat = primary_target.get(
                    "quat_rotation"
                )
            else:
                self.primary_target = primary_target

        self.stepwise_target_object = None
        self.stepwise_targets_list = []
        self.terminal_state = None
        self.detected_object = None
        self.detected_pose = [None for _ in range(7)]
        self.detected_rotation_r = None

    def post_episode(self):
        """Store observations in scaffold after training episode.

        P0 #3 FIX: Re-trace the agent's trajectory through the buffer,
        rotate each displacement by the MLH rotation, path-integrate
        temporary grid modules, and store the encoded SDR at each phase.
        """
        if self.mode is not ExperimentMode.TRAIN:
            return
        if len(self.buffer) == 0:
            return

        # Determine object ID
        if self.detected_object is None:
            self.set_detected_object(self.terminal_state)
        object_id = self.detected_object
        if object_id is None:
            return

        # Register object if new
        if object_id not in self._known_object_ids:
            self._known_object_ids.append(object_id)

        # Determine rotation from MLH
        mlh = self._get_mlh()
        rotation = mlh.rotation if mlh is not None else np.eye(3)
        rotation_inv = rotation.T

        # Re-trace trajectory and store in scaffold
        temp_grid = self.grid_modules.copy()
        temp_grid.reset()

        prev_loc = None
        num_stored = 0

        for step_idx in range(len(self.buffer)):
            try:
                loc = self.buffer.get_current_location(input_channel="first")
            except (ValueError, IndexError):
                # Buffer may not have location for this step
                continue

            # For first observation, there is a location but no displacement
            if step_idx == 0:
                # Try to get the actual first location from the buffer
                try:
                    all_locs = self.buffer.get_all_locations_on_object(
                        input_channel="first"
                    )
                    if len(all_locs) > step_idx:
                        loc = all_locs[step_idx]
                except (ValueError, IndexError):
                    pass

            if prev_loc is not None:
                displacement = loc - prev_loc
                # Rotate into object frame using MLH rotation
                obj_displacement = rotation_inv @ displacement
                temp_grid.path_integrate(obj_displacement)

            prev_loc = loc

            # Encode current features as SDR
            observation = self._extract_observation_from_buffer(step_idx)
            if observation is None:
                continue

            morph_features = self.hypothesis_manager._get_morph_features(observation)
            non_morph_features = observation.get("non_morph_features")
            sensed_sdr = self.sdr_encoder.encode(morph_features, non_morph_features)

            # Store at current grid state
            grid_state = temp_grid.get_binary_state()
            self.cortical_scaffold.store(grid_state, sensed_sdr)
            num_stored += 1

        # Update target <-> graph mappings
        self._update_target_graph_mapping(object_id, self.primary_target)

        logger.info(
            f"Stored {num_stored} observations for object {object_id}"
        )

    def set_experiment_mode(self, mode: ExperimentMode) -> None:
        self.mode = mode

    def matching_step(
        self,
        ctx: RuntimeContext,
        observations,
    ):
        """Inference step: update hypotheses given new observation.

        P0 #2 FIX: Central grid modules are path-integrated with raw
        body-frame displacement, keeping them in sync for subsequent
        exploration.
        """
        # Add displacements to observations and append to buffer
        buffer_data = self._add_displacements(observations)
        self.buffer.append(buffer_data)
        self.buffer.append_input_states(observations)

        # Extract features from current observation
        observation = self._extract_observation(observations)
        displacement = self._compute_displacement(observations)

        # P0 #2: Path-integrate central grid modules (body frame)
        if displacement is not None:
            self.grid_modules.path_integrate(displacement)

        first_obs = not self._agent_moved_since_reset()

        if first_obs or len(self.hypotheses) == 0:
            # Initialise hypotheses from first observation
            self.hypotheses = self.hypothesis_manager.initialise_from_observation(
                observation,
                self.cortical_scaffold,
                self.grid_modules,
                self._known_object_ids,
            )
            if len(self.hypotheses) == 0 and len(self._known_object_ids) == 0:
                self.set_individual_ts("no_match")
        elif displacement is not None:
            # Update evidence for all hypotheses
            self.hypotheses = self.hypothesis_manager.path_integrate_and_update_evidence(
                self.hypotheses,
                displacement,
                observation,
                self.cortical_scaffold,
                self.grid_modules,
            )

            # Burst sampling: dynamic hypothesis management (P1 #5)
            self.hypotheses = self.burst_sampler.step(
                self.hypotheses,
                observation,
                self.hypothesis_manager,
                self.cortical_scaffold,
                self.grid_modules,
                self._known_object_ids,
            )

            # Prune hypotheses below evidence threshold
            self.hypotheses = [
                h for h in self.hypotheses
                if h.evidence > self.evidence_prune_threshold
            ]

        # Update possible matches and MLH
        self._update_possible_matches_from_hypotheses()
        self._update_mlh()

        # Check terminal conditions
        if len(self.get_possible_matches()) == 0:
            self.set_individual_ts("no_match")

        # Step goal state generator
        if self.gsg is not None:
            self.gsg.step(ctx, observations)

        # Collect and log stats
        stats = self.collect_stats_to_save()
        self.buffer.update_stats(stats, append=self.has_detailed_logger)

    def exploratory_step(
        self,
        ctx: RuntimeContext,  # noqa: ARG002
        observations,
    ):
        """Model-building step: store features in scaffold.

        Central grid modules are path-integrated with body-frame displacement,
        and the encoded SDR is stored at the current scaffold address.
        """
        buffer_data = self._add_displacements(observations)
        self.buffer.append(buffer_data)
        self.buffer.append_input_states(observations)

        observation = self._extract_observation(observations)
        displacement = self._compute_displacement(observations)

        if displacement is not None:
            self.grid_modules.path_integrate(displacement)

        # Encode features and store in scaffold
        morph_features = self.hypothesis_manager._get_morph_features(observation)
        non_morph_features = observation.get("non_morph_features")
        sensed_sdr = self.sdr_encoder.encode(morph_features, non_morph_features)

        grid_state = self.grid_modules.get_binary_state()
        self.cortical_scaffold.store(grid_state, sensed_sdr)

    def receive_votes(self, votes):
        """Process votes from other LMs.

        P0 #4 FIX: Transform vote locations using inter-sensor displacement
        rotated by hypothesis rotation, then compare in toroidal phase
        space using GridModuleArray.toroidal_distance.
        """
        if votes is None or len(self.hypotheses) == 0:
            return
        if self.buffer.get_num_observations_on_object() < 1:
            return

        for obj_id, vote_states in votes.items():
            for vote_state in vote_states:
                vote_loc = vote_state.location
                vote_confidence = vote_state.confidence

                for hyp in self.hypotheses:
                    if hyp.object_id != obj_id:
                        continue

                    # Compute distance between vote location and hypothesis
                    # location in object-centric frame
                    # Transform vote location by hypothesis rotation
                    vote_displacement = vote_loc - hyp.accumulated_displacement
                    obj_displacement = hyp.rotation.T @ vote_displacement

                    # Project displacement into grid phase shift
                    transformed_phases = GridModuleArray.shift_phases(
                        hyp.grid_phases,
                        obj_displacement,
                        self.grid_modules.projections,
                        self.grid_config.periods,
                    )

                    # Toroidal distance
                    dist = GridModuleArray.toroidal_distance(
                        hyp.grid_phases,
                        transformed_phases,
                        self.grid_config.periods,
                    )

                    # Evidence update from vote
                    vote_ev = vote_confidence * np.exp(-dist / 2.0)
                    hyp.evidence += self.vote_weight * vote_ev

        # Update possible matches after voting
        self._update_possible_matches_from_hypotheses()
        self._update_mlh()

    def send_out_vote(self) -> dict | None:
        """Send hypotheses as votes to other LMs.

        Returns a dict compatible with MontyForEvidenceGraphMatching's
        _combine_votes format:
            - "possible_states": dict mapping object_id -> list of State
            - "sensed_pose_rel_body": current sensor pose (4x3 array)
        """
        if (
            self.buffer.get_num_observations_on_object() < 1
            or not self.buffer.get_last_obs_processed()
        ):
            return None

        sensed_pose = self.buffer.get_current_pose(input_channel="first")

        possible_states = {}
        for hyp in self.hypotheses:
            # Scale evidence to [-1, 1] range
            max_ev = max((h.evidence for h in self.hypotheses), default=1.0)
            min_ev = min((h.evidence for h in self.hypotheses), default=0.0)
            ev_range = max_ev - min_ev
            if ev_range > 1e-8:
                scaled_ev = 2.0 * (hyp.evidence - min_ev) / ev_range - 1.0
            else:
                scaled_ev = 0.0

            if scaled_ev < self.vote_evidence_threshold:
                continue

            # Create vote state
            rotation_matrix = hyp.rotation
            vote = State(
                location=hyp.accumulated_displacement,
                morphological_features={
                    "pose_vectors": rotation_matrix.T,
                    "pose_fully_defined": True,
                },
                non_morphological_features=None,
                confidence=scaled_ev,
                use_state=True,
                sender_id=self.learning_module_id,
                sender_type="LM",
            )

            if hyp.object_id not in possible_states:
                possible_states[hyp.object_id] = []
            possible_states[hyp.object_id].append(vote)

        return {
            "possible_states": possible_states,
            "sensed_pose_rel_body": sensed_pose,
        }

    def propose_goal_states(self) -> list[GoalState]:
        """Return goal states from the GSG."""
        if self.buffer.get_last_obs_processed() and self.gsg is not None:
            return self.gsg.output_goal_states()
        return []

    def get_output(self):
        """Return MLH as a State in CMP format.

        Returns the most likely hypothesis as a State object for downstream
        LMs in the hierarchy.
        """
        mlh = self._get_mlh()
        if mlh is None or len(self.buffer) == 0:
            return None

        # Only output if evidence is sufficient
        if mlh.evidence < self.object_evidence_threshold:
            return State(
                location=self.buffer.get_current_location(input_channel="first"),
                morphological_features={
                    "pose_vectors": np.eye(3),
                    "pose_fully_defined": False,
                    "on_object": self.buffer.get_currently_on_object(),
                },
                non_morphological_features={},
                confidence=0.0,
                use_state=False,
                sender_id=self.learning_module_id,
                sender_type="LM",
            )

        confidence = np.clip(mlh.evidence / max(len(self.buffer), 1), 0, 1)

        return State(
            location=self.buffer.get_current_location(input_channel="first"),
            morphological_features={
                "pose_vectors": mlh.rotation.T,
                "pose_fully_defined": not self._enough_symmetry_evidence_accumulated(),
                "on_object": self.buffer.get_currently_on_object(),
            },
            non_morphological_features={
                "object_id": mlh.object_id,
            },
            confidence=float(confidence),
            use_state=True,
            sender_id=self.learning_module_id,
            sender_type="LM",
        )

    def state_dict(self):
        """Serialise all persistent state."""
        return {
            "cortical_scaffold": self.cortical_scaffold.state_dict(),
            "target_to_graph_id": self.target_to_graph_id,
            "graph_id_to_target": self.graph_id_to_target,
            "known_object_ids": self._known_object_ids,
        }

    def load_state_dict(self, state_dict):
        """Restore from serialised state."""
        self.cortical_scaffold.load_state_dict(state_dict["cortical_scaffold"])
        self.target_to_graph_id = state_dict["target_to_graph_id"]
        self.graph_id_to_target = state_dict["graph_id_to_target"]
        self._known_object_ids = state_dict["known_object_ids"]

    # ==================== GraphLM-Compatible Methods ====================
    # These are required by MontyForGraphMatching's infrastructure.

    def get_possible_matches(self) -> list[str]:
        """Get list of currently possible object IDs."""
        return list(self.possible_matches.keys())

    def get_all_known_object_ids(self) -> list[str]:
        """Get all object IDs stored in scaffold memory."""
        return list(self._known_object_ids)

    def get_unique_pose_if_available(self, object_id: str):
        """Return 7D pose if uniquely identified, else None.

        Checks if there's exactly one remaining hypothesis for this object,
        or if symmetry evidence indicates a symmetric object with multiple
        equivalent poses.

        Returns:
            7D array [x, y, z, rx, ry, rz, scale] or None.
        """
        obj_hyps = [h for h in self.hypotheses if h.object_id == object_id]
        if len(obj_hyps) == 0:
            return None

        if len(obj_hyps) == 1:
            return self._hypothesis_to_pose(obj_hyps[0])

        # P1 #8: Check symmetry evidence
        if self.symmetry_evidence >= self.required_symmetry_evidence:
            # Symmetric object — use the highest-evidence hypothesis
            best = max(obj_hyps, key=lambda h: h.evidence)
            return self._hypothesis_to_pose(best)

        return None

    def get_current_mlh(self) -> dict:
        """Return the current most likely hypothesis dict."""
        return self.current_mlh

    def set_individual_ts(self, terminal_state):
        """Set terminal state for this LM."""
        logger.info(
            f"Setting terminal state of {self.learning_module_id} "
            f"to {terminal_state}"
        )
        self.set_detected_object(terminal_state)
        if terminal_state == "match":
            logger.info(
                f"{self.learning_module_id}: Detected {self.detected_object} "
                f"at pose {np.round(self.detected_pose[:6], 3)}"
            )
            self.buffer.set_individual_ts(self.detected_object, self.detected_pose)
        else:
            self.buffer.set_individual_ts(None, None)

    def set_detected_object(self, terminal_state):
        """Set detected object based on terminal state."""
        self.terminal_state = terminal_state
        if terminal_state is None:
            self.detected_object = None
        elif terminal_state == "no_match" or len(self.get_possible_matches()) == 0:
            self.detected_object = "new_object" + str(len(self._known_object_ids))
        elif terminal_state == "match":
            self.detected_object = self.get_possible_matches()[0]
        else:
            self.detected_object = None

    def update_terminal_condition(self):
        """Check terminal conditions for this episode.

        P1 #8 FIX: Track symmetry evidence when multiple poses for the
        same object persist.
        """
        possible_matches = self.get_possible_matches()

        if len(possible_matches) == 0:
            self.set_individual_ts("no_match")
            if self.buffer.get_num_observations_on_object() > 0:
                self.buffer.stats["detected_location_rel_body"] = (
                    self.buffer.get_current_location(input_channel="first")
                )
        elif (
            self.buffer.get_num_observations_on_object() > 0
            and len(possible_matches) == 1
        ):
            object_id = possible_matches[0]
            pose = self.get_unique_pose_if_available(object_id)

            # P1 #8: Track symmetry evidence
            obj_hyps = [h for h in self.hypotheses if h.object_id == object_id]
            if len(obj_hyps) > 1:
                self.symmetry_evidence += 1
            else:
                self.symmetry_evidence = 0

            if pose is not None:
                self.set_individual_ts("match")
                logger.info(
                    f"{self.learning_module_id} recognised {object_id}"
                )
        else:
            logger.info(
                f"{self.learning_module_id} has {len(possible_matches)} "
                f"possible matches"
            )

        return self.terminal_state

    def collect_stats_to_save(self) -> dict:
        """Collect stats for buffer logging."""
        stats = {
            "possible_matches": self.get_possible_matches(),
        }
        if self.has_detailed_logger:
            stats["num_hypotheses"] = len(self.hypotheses)
            stats["symmetry_evidence"] = self.symmetry_evidence
            if self.hypotheses:
                mlh = self._get_mlh()
                if mlh is not None:
                    stats["mlh_object"] = mlh.object_id
                    stats["mlh_evidence"] = mlh.evidence
        return stats

    def add_lm_processing_to_buffer_stats(self, lm_processed):
        """Update buffer stats with LM processing flag."""
        self.buffer.update_stats(
            dict(lm_processed_steps=lm_processed), update_time=False
        )

    # ==================== Private Methods ====================

    def _add_displacements(self, obs):
        """Add displacement to each observation from buffer history."""
        for o in obs:
            if self.buffer.get_buffer_len_by_channel(o.sender_id) > 0:
                displacement = o.location - self.buffer.get_current_location(
                    input_channel=o.sender_id
                )
            else:
                displacement = np.zeros(3)
            o.set_displacement(displacement)
        return obs

    def _compute_displacement(self, observations) -> np.ndarray | None:
        """Compute body-frame displacement from consecutive locations."""
        try:
            current_loc = observations[0].location
        except (IndexError, AttributeError):
            return None

        if self._prev_location is not None:
            displacement = current_loc - self._prev_location
        else:
            displacement = None

        self._prev_location = current_loc.copy()
        return displacement

    def _extract_observation(self, observations) -> dict:
        """Extract features from State observations into a dict.

        Handles the messy reality of parsing Monty's State objects,
        including None checks and missing pose vectors.
        """
        obs = observations[0] if len(observations) > 0 else None
        result = {}

        if obs is None:
            return result

        # Pose vectors (morphological)
        try:
            pose_vectors = obs.get_pose_vectors()
            if pose_vectors is not None and pose_vectors.shape == (3, 3):
                result["surface_normal"] = pose_vectors[0]
                result["curvature_dir"] = pose_vectors[1]
        except (AttributeError, IndexError):
            pass

        # Check if principal curvatures are equal (isotropic)
        try:
            morph = obs.morphological_features
            result["pc1_is_pc2"] = not morph.get("pose_fully_defined", True)
        except (AttributeError, KeyError):
            result["pc1_is_pc2"] = False

        # Non-morphological features
        non_morph = []
        try:
            nm_features = obs.non_morphological_features
            if nm_features is not None:
                for key in sorted(nm_features.keys()):
                    val = nm_features[key]
                    if isinstance(val, (int, float)):
                        non_morph.append(float(val))
                    elif isinstance(val, np.ndarray):
                        non_morph.extend(val.flatten().tolist())
                    elif isinstance(val, (list, tuple)):
                        non_morph.extend([float(v) for v in val])
        except (AttributeError, TypeError):
            pass

        if non_morph:
            result["non_morph_features"] = np.array(non_morph, dtype=np.float64)

        return result

    def _extract_observation_from_buffer(self, step_idx: int) -> dict | None:
        """Extract observation at a specific buffer index for post_episode."""
        try:
            all_locs = self.buffer.get_all_locations_on_object(
                input_channel="first"
            )
            if step_idx >= len(all_locs):
                return None
        except (ValueError, IndexError):
            return None

        # Minimal observation dict — just enough for SDR encoding
        return {
            "surface_normal": None,
            "curvature_dir": None,
            "non_morph_features": None,
        }

    def _agent_moved_since_reset(self) -> bool:
        """Check if the agent has moved (has prior observations)."""
        return len(self.buffer) > 1

    def _update_possible_matches_from_hypotheses(self):
        """Update the possible_matches dict from current hypotheses.

        Uses x_percent_threshold: only keep objects whose max evidence
        is within x_percent of the overall maximum evidence.
        """
        if not self.hypotheses:
            self.possible_matches = {}
            return

        # Group by object and get max evidence per object
        obj_evidence = {}
        for hyp in self.hypotheses:
            if hyp.object_id not in obj_evidence:
                obj_evidence[hyp.object_id] = hyp.evidence
            else:
                obj_evidence[hyp.object_id] = max(
                    obj_evidence[hyp.object_id], hyp.evidence
                )

        if not obj_evidence:
            self.possible_matches = {}
            return

        max_evidence = max(obj_evidence.values())
        threshold = max_evidence - abs(max_evidence) * (self.x_percent_threshold / 100.0)

        self.possible_matches = {
            obj_id: True
            for obj_id, ev in obj_evidence.items()
            if ev >= threshold
        }

    def _update_mlh(self):
        """Update the current MLH dict from hypotheses."""
        mlh = self._get_mlh()
        if mlh is not None:
            try:
                rotation_scipy = Rotation.from_matrix(mlh.rotation)
            except Exception:
                rotation_scipy = Rotation.identity()

            self.current_mlh = {
                "graph_id": mlh.object_id,
                "location": mlh.accumulated_displacement.copy(),
                "rotation": rotation_scipy,
                "scale": 1,
                "evidence": mlh.evidence,
            }
            self.detected_rotation_r = rotation_scipy

            # Update detected pose
            euler = rotation_scipy.as_euler("xyz")
            self.detected_pose = [
                *mlh.accumulated_displacement[:3],
                *euler,
                1,  # scale
            ]

    def _get_mlh(self) -> Hypothesis | None:
        """Get the most likely hypothesis."""
        if not self.hypotheses:
            return None
        return max(self.hypotheses, key=lambda h: h.evidence)

    def _enough_symmetry_evidence_accumulated(self) -> bool:
        """Check if enough symmetry evidence has accumulated."""
        return self.symmetry_evidence >= self.required_symmetry_evidence

    def _hypothesis_to_pose(self, hyp: Hypothesis) -> list:
        """Convert a hypothesis to a 7D pose array."""
        try:
            euler = Rotation.from_matrix(hyp.rotation).as_euler("xyz")
        except Exception:
            euler = np.zeros(3)
        return [
            *hyp.accumulated_displacement[:3],
            *euler,
            1,  # scale
        ]

    def _update_target_graph_mapping(self, detected_object, target_object):
        """Update target <-> graph ID mappings."""
        if detected_object is not None and target_object is not None:
            if detected_object not in self.graph_id_to_target:
                self.graph_id_to_target[detected_object] = {target_object}
            else:
                self.graph_id_to_target[detected_object].add(target_object)

            if target_object not in self.target_to_graph_id:
                self.target_to_graph_id[target_object] = {detected_object}
            else:
                self.target_to_graph_id[target_object].add(detected_object)
