# Copyright 2025-2026  Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import quaternion as qt

from tbp.monty.cmp import Message
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.abstract_monty_classes import (
    SensorModule,
    SensorObservation,
)
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    SensorState,
)
from tbp.monty.frameworks.models.sensor_modules import (
    DefaultMessageNoise,
    FeatureChangeFilter,
    MessageNoise,
    NoMessageNoise,
    ObservationProcessor,
    PassthroughPerceptFilter,
    PerceptFilter,
    SnapshotTelemetry,
)
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.frameworks.utils.edge_detection import (
    EdgeDetector,
)
from tbp.monty.frameworks.utils.sensor_processing import (
    arc_length_corrected_displacement,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import (
    TangentFrame,
    project_onto_tangent_plane,
)
from tbp.monty.math import DEFAULT_TOLERANCE

logger = logging.getLogger(__name__)


class TwoDSensorModule(SensorModule):
    """Sensor Module that extracts edges and other features at 2D locations.

    Extends the base sensor module to detect edges on an object's surface (e.g.
    edges of a logo on a cup) and enable an associated LM to build a
    corresponding 2D model of the 3D object. Movements in 2D are estimated by
    unrolling the 3D surface into a
    local tangent plane, which requires extracting surface normals and principal
    curvatures at each observation point to perform the projection correctly.

    The 2D position is initialized to the world x,y coordinates. A 2D model is
    built by accumulating tangent plane displacements.
    """

    def __init__(
        self,
        sensor_module_id: str,
        features: list[str],
        save_raw_obs: bool = False,
        pc1_is_pc2_threshold: int = 10,
        is_surface_sm: bool = False,
        edge_detector: EdgeDetector | None = None,
        noise_params: dict[str, Any] | None = None,
        delta_thresholds: dict[str, Any] | None = None,
    ):
        """Initialize 2D Sensor Module.

        Args:
            sensor_module_id: Name of sensor module.
            features: Which features to extract.
            save_raw_obs: Whether to save raw sensory input for logging.
            pc1_is_pc2_threshold: Maximum difference between pc1 and pc2 to be
                classified as being roughly the same (ignore curvature directions).
                Defaults to 10.
            is_surface_sm: Surface SMs do not require that the central pixel is
                "on object" in order to process the observation (i.e., extract
                features). Defaults to False.
            edge_detector: Feature extractor for edges.
            noise_params: Dictionary of noise amount for each feature.
            delta_thresholds: If given, a FeatureChangeFilter will be used to
                check whether the current state's features are significantly different
                from the previous with tolerances set according to `delta_thresholds`.
                Defaults to None.
        """
        self._observation_processor = ObservationProcessor(
            features=features,
            sensor_module_id=sensor_module_id,
            pc1_is_pc2_threshold=pc1_is_pc2_threshold,
            is_surface_sm=is_surface_sm,
        )
        if noise_params:
            self._message_noise: MessageNoise = DefaultMessageNoise(
                noise_params=noise_params
            )
        else:
            self._message_noise = NoMessageNoise()
        if delta_thresholds:
            self._percept_filter: PerceptFilter = FeatureChangeFilter(
                delta_thresholds=delta_thresholds
            )
        else:
            self._percept_filter = PassthroughPerceptFilter()
        self._snapshot_telemetry = SnapshotTelemetry()

        self.features = features
        self.processed_obs = []
        self.states = []
        self.sensor_module_id = sensor_module_id
        self.save_raw_obs = save_raw_obs
        self.edge_detector = edge_detector

        self._previous_3d_location: np.ndarray | None = None
        self._previous_2d_location: np.ndarray = np.zeros(2)
        self._tangent_frame: TangentFrame | None = None

    def pre_episode(self) -> None:
        self._snapshot_telemetry.reset()
        self._percept_filter.reset()
        self.is_exploring = False
        self.processed_obs = []
        self.states = []
        self._previous_3d_location = None
        self._previous_2d_location = np.zeros(2)
        self._tangent_frame = None

    def update_state(self, agent: AgentState):
        """Update information about the sensor's location and rotation."""
        sensor = agent.sensors[SensorID(self.sensor_module_id)]
        self.state = SensorState(
            position=agent.position
            + qt.rotate_vectors(agent.rotation, sensor.position),
            rotation=agent.rotation * sensor.rotation,
        )

    def state_dict(self):
        state_dict = self._snapshot_telemetry.state_dict()
        state_dict.update(processed_observations=self.processed_obs)
        return state_dict

    def step(
        self,
        ctx: RuntimeContext,
        observation: SensorObservation,
        motor_only_step: bool = False,
    ) -> Message:
        """Turn raw observations into dict of features at location.

        Args:
            ctx: The runtime context.
            observation: Raw observations.
            motor_only_step: If True, mark the resulting Message as not to be
                passed to an LM.

        Returns:
            Message with features and morphological features. Noise may be added.
            use_state flag may be set.
        """
        if self.save_raw_obs and not self.is_exploring:
            self._snapshot_telemetry.raw_observation(
                observation, self.state.rotation, self.state.position
            )

        observed_state = self._observation_processor.process(observation)

        curvature_pose_vectors = observed_state.get_pose_vectors().copy()
        true_surface_normal = observed_state.get_surface_normal().copy()

        # Only edges define pose for 2D sensor; reset curvature-based flag.
        observed_state.morphological_features["pose_fully_defined"] = False

        if observed_state.use_state and observed_state.get_on_object():
            observed_state = self._extract_2d_edge(
                observed_state,
                observation,
            )

        if observed_state.use_state:
            observed_state = self._message_noise(observed_state, rng=ctx.rng)

        if motor_only_step:
            # Set interesting-features flag to False, as should not be passed to
            # LM, even in e.g. pre-training experiments that might otherwise do so
            observed_state.use_state = False

        observed_state = self._update_2d_position_and_displacement(
            observed_state, curvature_pose_vectors, true_surface_normal
        )

        observed_state = self._percept_filter(observed_state)

        if not self.is_exploring:
            self.processed_obs.append(observed_state.__dict__)
            self.states.append(self.state)

        return observed_state

    def _extract_2d_edge(
        self,
        state: Message,
        observation: SensorObservation,
    ) -> Message:
        """Extract 2D edge-based pose if edge is detected.

        This method attempts to create a fully-defined pose (normal + 2 tangents)
        using edge detection, replacing the standard curvature-based tangents.

        Args:
            state: Message with standard features from ObservationProcessor
            observation: Sensor observation.

        Returns:
            Message with edge-based pose vectors if edge detected,
            otherwise returns the original state unchanged.
        """
        edge = self.edge_detector(observation)

        if not edge.has_edge or (edge.strength and edge.is_geometric_edge):
            return state

        state.morphological_features["pose_vectors"] = edge.pose_2d
        state.morphological_features["pose_fully_defined"] = True

        if "edge_strength" in self.features:
            state.non_morphological_features["edge_strength"] = edge.strength
        if "coherence" in self.features:
            state.non_morphological_features["coherence"] = edge.coherence

        return state

    def _update_2d_position_and_displacement(
        self,
        observed_state: Message,
        pose_3d: np.ndarray | None,
        surface_normal_3d: np.ndarray | None,
    ) -> Message:
        """Project the 3D step onto the tangent plane to get a 2D displacement.

        Args:
            observed_state: Message to update with 2D displacement and position.
            pose_3d: Real pose vectors based on 3D object used for
                arc-length correction. If None, chord length is used as-is.
            surface_normal_3d: True surface normal from curvature estimation,
                saved before edge detection may overwrite pose_vectors.

        Returns:
            The updated state with 2D position and displacement.
        """
        if not observed_state.get_on_object():
            observed_state.set_displacement(np.zeros(3))
            return observed_state

        current_3d_location = observed_state.location.copy()

        if self._previous_3d_location is None or surface_normal_3d is None:
            self._previous_3d_location = current_3d_location
            self._previous_2d_location = current_3d_location[:2].copy()
            observed_state.location = np.array(
                # Setting z = 0 assumes that camera is TODO
                [current_3d_location[0], current_3d_location[1], 0.0]
            )
            if surface_normal_3d is not None:
                self._tangent_frame = TangentFrame(surface_normal_3d)
            return observed_state

        displacement_3d = current_3d_location - self._previous_3d_location
        d_tan = project_onto_tangent_plane(displacement_3d, surface_normal_3d)

        if np.linalg.norm(d_tan) < DEFAULT_TOLERANCE:
            self._previous_3d_location = current_3d_location
            observed_state.set_displacement(np.zeros(3))
            observed_state.location = np.array(
                # See previous comment on setting z = 0
                [
                    self._previous_2d_location[0],
                    self._previous_2d_location[1],
                    0.0,
                ]
            )
            return observed_state

        du = np.dot(d_tan, self._tangent_frame.basis_u)
        dv = np.dot(d_tan, self._tangent_frame.basis_v)

        principal_curvatures = observed_state.morphological_features.get(
            "principal_curvatures"
        )

        if principal_curvatures is not None and pose_3d is not None:
            du, dv = arc_length_corrected_displacement(
                du,
                dv,
                self._tangent_frame.basis_u,
                self._tangent_frame.basis_v,
                principal_curvatures,
                pose_3d,
            )

        self._previous_2d_location += [du, dv]
        observed_state.set_displacement(np.array([du, dv, 0.0]))
        observed_state.location = np.array(
            [
                self._previous_2d_location[0],
                self._previous_2d_location[1],
                0.0,
            ]
        )
        return observed_state
