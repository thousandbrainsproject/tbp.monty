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

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Protocol, TypedDict

import numpy as np
import quaternion
from scipy.spatial.transform import Rotation
from skimage.color import rgb2hsv

from tbp.monty.frameworks.models.abstract_monty_classes import SensorModule
from tbp.monty.frameworks.models.states import State
from tbp.monty.frameworks.utils.sensor_processing import (
    log_sign,
    principal_curvatures,
    scale_clip,
    surface_normal_naive,
    surface_normal_ordinary_least_squares,
    surface_normal_total_least_squares,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import get_angle

logger = logging.getLogger(__name__)


class SnapshotTelemetry:
    """Keeps track of raw observation snapshot telemetry."""

    def __init__(self):
        self.poses = []
        self.raw_observations = []

    def reset(self):
        """Reset the snapshot telemetry."""
        self.raw_observations = []
        self.poses = []

    def raw_observation(
        self, raw_observation, rotation: quaternion.quaternion, position: np.ndarray
    ):
        """Record a snapshot of a raw observation and its pose information.

        Args:
            raw_observation: Raw observation.
            rotation: Rotation of the sensor.
            position: Position of the sensor.
        """
        self.raw_observations.append(raw_observation)
        self.poses.append(
            dict(
                sm_rotation=quaternion.as_float_array(rotation),
                sm_location=np.array(position),
            )
        )

    def state_dict(self) -> dict[str, list[np.ndarray]]:
        """Returns recorded raw observation snapshots.

        Returns:
            Dictionary containing a list of raw observations in `raw_observations` and
            a list of pose information for each observation in `sm_properties`.
        """
        assert len(self.poses) == len(self.raw_observations), (
            "Each raw observation should have a corresponding pose information."
        )
        return dict(raw_observations=self.raw_observations, sm_properties=self.poses)


class HabitatObservation(TypedDict):
    semantic_3d: np.ndarray
    sensor_frame_data: np.ndarray
    world_camera: np.ndarray
    rgba: np.ndarray
    depth: np.ndarray


class SurfaceNormalMethod(Enum):
    TLS = "TLS"
    """Total Least-Squares"""
    OLS = "OLS"
    """Ordinary Least-Squares"""
    NAIVE = "naive"
    """Naive"""


@dataclass
class HabitatObservationProcessorTelemetry:
    processed_obs: State
    visited_loc: Any
    visited_normal: Any | None


class HabitatObservationProcessor:
    """Processes Habitat observations into a Cortical Message."""

    CURVATURE_FEATURES: ClassVar[list[str]] = [
        "principal_curvatures",
        "principal_curvatures_log",
        "gaussian_curvature",
        "mean_curvature",
        "gaussian_curvature_sc",
        "mean_curvature_sc",
        "curvature_for_TM",
    ]

    POSSIBLE_FEATURES: ClassVar[list[str]] = [
        "on_object",
        "object_coverage",
        "min_depth",
        "mean_depth",
        "rgba",
        "hsv",
        "pose_vectors",
        "principal_curvatures",
        "principal_curvatures_log",
        "pose_fully_defined",
        "gaussian_curvature",
        "mean_curvature",
        "gaussian_curvature_sc",
        "mean_curvature_sc",
        "curvature_for_TM",
        "coords_for_TM",
    ]

    def __init__(
        self,
        features: list[str],
        sensor_module_id: str,
        pc1_is_pc2_threshold=10,
        surface_normal_method=SurfaceNormalMethod.TLS,
        weight_curvature=True,
        is_surface_sm=False,
    ) -> None:
        """Initializes the HabitatObservationProcessor.

        Args:
            features: List of features to extract.
            pc1_is_pc2_threshold: Maximum difference between pc1 and pc2 to be
                classified as being roughly the same (ignore curvature directions).
                Defaults to 10.
            sensor_module_id: ID of sensor module.
            surface_normal_method: Method to use for surface normal extraction. Defaults
              to TLS.
            weight_curvature: Whether to use the weighted implementation for principal
                curvature extraction (True) or unweighted (False). Defaults to True.
            is_surface_sm: Surface SMs do not require that the central pixel is
                "on object" in order to process the observation (i.e., extract
                features). Defaults to False.
        """
        for feature in features:
            assert feature in self.POSSIBLE_FEATURES, (
                f"{feature} not part of {self.POSSIBLE_FEATURES}"
            )
        self._features = features
        self._is_surface_sm = is_surface_sm
        self._pc1_is_pc2_threshold = pc1_is_pc2_threshold
        self._sensor_module_id = sensor_module_id
        self._surface_normal_method = surface_normal_method
        self._weight_curvature = weight_curvature

    def process(
        self, observation: HabitatObservation
    ) -> tuple[State, HabitatObservationProcessorTelemetry]:
        """Processes observation.

        Args:
            observation: Habitat observation.

        Returns:
            Cortical Message.
        """
        obs_3d = observation["semantic_3d"]
        sensor_frame_data = observation["sensor_frame_data"]
        world_camera = observation["world_camera"]
        rgba_feat = observation["rgba"]
        depth_feat = (
            observation["depth"]
            .reshape(observation["depth"].size, 1)
            .astype(np.float64)
        )
        # Assuming squared patches
        center_row_col = rgba_feat.shape[0] // 2
        # Calculate center ID for flat semantic obs
        obs_dim = int(np.sqrt(obs_3d.shape[0]))
        half_obs_dim = obs_dim // 2
        center_id = half_obs_dim + obs_dim * half_obs_dim
        # Extract all specified features
        features = {}
        if "object_coverage" in self._features:
            # Last dimension is semantic ID (integer >0 if on any object)
            features["object_coverage"] = sum(obs_3d[:, 3] > 0) / len(obs_3d[:, 3])
            assert features["object_coverage"] <= 1.0, (
                "Coverage cannot be greater than 100%"
            )

        x, y, z, semantic_id = obs_3d[center_id]
        on_object = semantic_id > 0
        if on_object or (self._is_surface_sm and features["object_coverage"] > 0):
            (
                features,
                morphological_features,
                invalid_signals,
            ) = self._extract_and_add_features(
                features,
                obs_3d,
                rgba_feat,
                depth_feat,
                center_id,
                center_row_col,
                sensor_frame_data,
                world_camera,
            )
        else:
            invalid_signals = True
            morphological_features = {}

        if "on_object" in self._features:
            morphological_features["on_object"] = float(on_object)

        # Sensor module returns features at a location in the form of a State class.
        # use_state is a bool indicating whether the input is "interesting",
        # which indicates that it merits processing by the learning module; by default
        # it will always be True so long as the surface normal and principal curvature
        # directions were valid; certain SMs and policies used separately can also set
        # it to False under appropriate conditions

        observed_state = State(
            location=np.array([x, y, z]),
            morphological_features=morphological_features,
            non_morphological_features=features,
            confidence=1.0,
            use_state=on_object and not invalid_signals,
            sender_id=self._sensor_module_id,
            sender_type="SM",
        )
        # This is just for logging! Do not use _ attributes for matching
        observed_state._semantic_id = semantic_id

        telemetry = HabitatObservationProcessorTelemetry(
            processed_obs=observed_state,
            visited_loc=observed_state.location,
            visited_normal=morphological_features["pose_vectors"][0]
            if "pose_vectors" in morphological_features.keys()
            else None,
        )

        return observed_state, telemetry

    def _extract_and_add_features(
        self,
        features: dict[str, Any],
        obs_3d: np.ndarray,
        rgba_feat: np.ndarray,
        depth_feat: np.ndarray,
        center_id: int,
        center_row_col: int,
        sensor_frame_data: np.ndarray,
        world_camera: np.ndarray,
    ) -> tuple[dict[str, Any], dict[str, Any], bool]:
        """Extract features configured for extraction from sensor patch.

        Returns the features in the patch, and True if the surface normal
        or principal curvature directions were ill-defined.

        Returns:
            features: The features in the patch.
            morphological_features: ?
            invalid_signals: True if the surface normal or principal curvature
                directions were ill-defined.
        """
        # ------------ Extract Morphological Features ------------
        # Get surface normal for graph matching with features
        surface_normal, valid_sn = self._get_surface_normals(
            obs_3d, sensor_frame_data, center_id, world_camera
        )

        k1, k2, dir1, dir2, valid_pc = principal_curvatures(
            obs_3d, center_id, surface_normal, weighted=self._weight_curvature
        )
        # TODO: test using log curvatures instead
        if np.abs(k1 - k2) < self._pc1_is_pc2_threshold:
            pose_fully_defined = False
        else:
            pose_fully_defined = True

        morphological_features: dict[str, Any] = {
            "pose_vectors": np.vstack(
                [
                    surface_normal,
                    dir1,
                    dir2,
                ]
            ),
            "pose_fully_defined": pose_fully_defined,
        }
        # ---------- Extract Optional, Non-Morphological Features ----------
        if "rgba" in self._features:
            features["rgba"] = rgba_feat[center_row_col, center_row_col]
        if "min_depth" in self._features:
            features["min_depth"] = np.min(depth_feat[obs_3d[:, 3] != 0])
        if "mean_depth" in self._features:
            features["mean_depth"] = np.mean(depth_feat[obs_3d[:, 3] != 0])
        if "hsv" in self._features:
            rgba = rgba_feat[center_row_col, center_row_col]
            hsv = rgb2hsv(rgba[:3])
            features["hsv"] = hsv

        # Note we only determine curvature if we could determine a valid surface normal
        if any(feat in self.CURVATURE_FEATURES for feat in self._features) and valid_sn:
            if valid_pc:
                # Only process the below features if the principal curvature was valid,
                # and therefore we have a defined k1, k2 etc.
                if "principal_curvatures" in self._features:
                    features["principal_curvatures"] = np.array([k1, k2])

                if "principal_curvatures_log" in self._features:
                    features["principal_curvatures_log"] = log_sign(np.array([k1, k2]))

                if "gaussian_curvature" in self._features:
                    features["gaussian_curvature"] = k1 * k2

                if "mean_curvature" in self._features:
                    features["mean_curvature"] = (k1 + k2) / 2

                if "gaussian_curvature_sc" in self._features:
                    gc = k1 * k2
                    gc_scaled_clipped = scale_clip(gc, 4096)
                    features["gaussian_curvature_sc"] = gc_scaled_clipped

                if "mean_curvature_sc" in self._features:
                    mc = (k1 + k2) / 2
                    mc_scaled_clipped = scale_clip(mc, 256)
                    features["mean_curvature_sc"] = mc_scaled_clipped
        else:
            # Flag that PC directions are non-meaningful for e.g. downstream motor
            # policies
            features["pose_fully_defined"] = False

        invalid_signals = (not valid_sn) or (not valid_pc)
        if invalid_signals:
            logger.debug("Either the surface-normal or pc-directions were ill-defined")

        return features, morphological_features, invalid_signals

    def _get_surface_normals(
        self,
        obs_3d: np.ndarray,
        sensor_frame_data: np.ndarray,
        center_id: int,
        world_camera: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        if self._surface_normal_method == SurfaceNormalMethod.TLS:
            surface_normal, valid_sn = surface_normal_total_least_squares(
                obs_3d, center_id, world_camera[:3, 2]
            )
        elif self._surface_normal_method == SurfaceNormalMethod.OLS:
            surface_normal, valid_sn = surface_normal_ordinary_least_squares(
                sensor_frame_data, world_camera, center_id
            )
        elif self._surface_normal_method == SurfaceNormalMethod.NAIVE:
            surface_normal, valid_sn = surface_normal_naive(
                obs_3d, patch_radius_frac=2.5
            )
        else:
            raise ValueError(
                f"surface_normal_method must be in [{SurfaceNormalMethod.TLS} (default)"
                f", {SurfaceNormalMethod.OLS}, {SurfaceNormalMethod.NAIVE}]."
            )

        return surface_normal, valid_sn


class Probe(SensorModule):
    """A probe that can be inserted into Monty in place of a sensor module.

    It will track raw observations for logging, and can be used by experiments
    for positioning procedures, visualization, etc.

    What distinguishes a probe from a sensor module is that it does not process
    observations and does not emit a Cortical Message.
    """

    def __init__(
        self,
        rng,  # noqa: ARG002
        sensor_module_id: str,
        save_raw_obs: bool,
    ):
        """Initialize the probe.

        Args:
            rng: Random number generator. Unused.
            sensor_module_id: Name of sensor module.
            save_raw_obs: Whether to save raw sensory input for logging.
        """
        super().__init__()

        self.is_exploring = False
        self.sensor_module_id = sensor_module_id
        self.state = None
        self.save_raw_obs = save_raw_obs

        self._snapshot_telemetry = SnapshotTelemetry()

    def state_dict(self):
        return self._snapshot_telemetry.state_dict()

    def update_state(self, state):
        """Update information about the sensors location and rotation."""
        # TODO: This stores the entire AgentState. Extract sensor-specific state.
        self.state = state

    def step(self, data) -> State | None:
        if self.save_raw_obs and not self.is_exploring:
            self._snapshot_telemetry.raw_observation(
                data,
                self.state["rotation"],
                self.state["location"]
                if "location" in self.state.keys()
                else self.state["position"],
            )

        return None

    def pre_episode(self):
        """Reset buffer and is_exploring flag."""
        self._snapshot_telemetry.reset()
        self.is_exploring = False


class MessageNoise(Protocol):
    def __call__(self, state: State) -> State: ...


def no_message_noise(state: State) -> State:
    """No noise function.

    Returns:
        State with no noise added.
    """
    return state


class DefaultMessageNoise(MessageNoise):
    def __init__(self, noise_params: dict[str, Any], rng):
        self.noise_params = noise_params
        self.rng = rng

    def __call__(self, state: State) -> State:
        """Add noise to features specified in noise_params.

        Noise params should have structure {"features":
                                                {"feature_keys": noise_amount, ...},
                                            "locations": noise_amount}
        noise_amount specifies the standard deviation of the gaussian noise sampled
        for real valued features. For boolian features it specifies the probability
        that the boolean flips.
        If we are dealing with normed vectors (surface_normal or curvature_directions)
        the noise is applied by rotating the vector given a sampled rotation. Otherwise
        noise is just added onto the perceived feature value.

        Args:
            state: State to add noise to.

        Returns:
            State with noise added.
        """
        if "features" in self.noise_params:
            for key in self.noise_params["features"]:
                if key in state.morphological_features:
                    if key == "pose_vectors":
                        # apply randomly sampled rotation to xyz axes with standard
                        # deviation specified in noise_params
                        # TODO: apply same rotation to both to make sure they stay
                        # orthogonal?
                        noise_angles = self.rng.normal(
                            0, self.noise_params["features"][key], 3
                        )
                        noise_rotation = Rotation.from_euler(
                            "xyz", noise_angles, degrees=True
                        )
                        state.morphological_features[key] = noise_rotation.apply(
                            state.morphological_features[key]
                        )
                    else:
                        state.morphological_features[key] = (
                            self.add_noise_to_feat_value(
                                feat_name=key,
                                feat_val=state.morphological_features[key],
                            )
                        )
                elif key in state.non_morphological_features:
                    state.non_morphological_features[key] = (
                        self.add_noise_to_feat_value(
                            feat_name=key,
                            feat_val=state.non_morphological_features[key],
                        )
                    )
        if "location" in self.noise_params:
            noise = self.rng.normal(0, self.noise_params["location"], 3)
            state.location = state.location + noise

        return state

    def add_noise_to_feat_value(self, feat_name, feat_val):
        if isinstance(feat_val, bool):
            # Flip boolean variable with probability specified in
            # noise_params
            if self.rng.random() < self.noise_params["features"][feat_name]:
                new_feat_val = not (feat_val)
            else:
                new_feat_val = feat_val

        else:
            # Add gaussian noise with standard deviation specified in
            # noise_params
            shape = feat_val.shape
            noise = self.rng.normal(0, self.noise_params["features"][feat_name], shape)
            new_feat_val = feat_val + noise
            if feat_name == "hsv":  # make sure hue stays in 0-1 range
                new_feat_val[0] = np.clip(new_feat_val[0], 0, 1)
        return new_feat_val


class HabitatSM(SensorModule):
    """Sensor Module that turns Habitat camera obs into features at locations.

    Takes in camera rgba and depth input and calculates locations from this.
    It also extracts features which are currently: on_object, rgba, surface_normal,
    curvature.
    """

    def __init__(
        self,
        rng,
        sensor_module_id: str,
        features: list[str],
        save_raw_obs: bool = False,
        pc1_is_pc2_threshold: int = 10,
        noise_params: dict[str, Any] | None = None,
        is_surface_sm: bool = False,
        delta_thresholds: dict[str, Any] | None = None,
    ) -> None:
        """Initialize Sensor Module.

        Args:
            rng: Random number generator.
            sensor_module_id: Name of sensor module.
            features: Which features to extract. In [on_object, rgba, surface_normal,
                principal_curvatures, curvature_directions, gaussian_curvature,
                mean_curvature]
            save_raw_obs: Whether to save raw sensory input for logging.
            pc1_is_pc2_threshold: Maximum difference between pc1 and pc2 to be
                classified as being roughly the same (ignore curvature directions).
                Defaults to 10.
            noise_params: Dictionary of noise amount for each feature.
            is_surface_sm: Surface SMs do not require that the central pixel is
                "on object" in order to process the observation
                (i.e., extract features). Defaults to False.
            delta_thresholds: If given, a FeatureChangeFilter will be used to
                check whether the current state's features are significantly different
                from the previous with tolerances set according to `delta_thresholds`.
                Defaults to None.

        Note:
            When using feature at location matching with graphs, surface_normal and
            on_object needs to be in the list of features.

        Note:
            gaussian_curvature and mean_curvature should be used together to contain
            the same information as principal_curvatures.
        """
        self._habitat_observation_processor = HabitatObservationProcessor(
            features=features,
            sensor_module_id=sensor_module_id,
            pc1_is_pc2_threshold=pc1_is_pc2_threshold,
            is_surface_sm=is_surface_sm,
        )
        if noise_params:
            self._message_noise: MessageNoise = DefaultMessageNoise(
                noise_params=noise_params, rng=rng
            )
        else:
            self._message_noise = no_message_noise
        if delta_thresholds:
            self._state_filter: StateFilter = FeatureChangeFilter(
                delta_thresholds=delta_thresholds
            )
        else:
            self._state_filter = PassthroughStateFilter()
        self._snapshot_telemetry = SnapshotTelemetry()
        # Tests check sm.features, not sure if this should be exposed
        self.features = features
        self.processed_obs = []
        self.states = []
        # TODO: give more descriptive & distinct names
        self.sensor_module_id = sensor_module_id
        self.save_raw_obs = save_raw_obs
        # Store visited locations in global environment coordinates to help inform
        # more intelligent motor-policies
        # TODO consider adding a flag or mixin to determine when these are actually
        # saved
        self.visited_locs = []
        self.visited_normals = []

    def pre_episode(self):
        """Reset buffer and is_exploring flag."""
        super().pre_episode()
        self._snapshot_telemetry.reset()
        self._state_filter.reset()
        self.is_exploring = False
        self.processed_obs = []
        self.states = []
        self.visited_locs = []
        self.visited_normals = []

    def update_state(self, state):
        """Update information about the sensors location and rotation."""
        agent_position = state["position"]
        sensor_position = state["sensors"][self.sensor_module_id + ".rgba"]["position"]
        if "motor_only_step" in state.keys():
            self.motor_only_step = state["motor_only_step"]
        else:
            self.motor_only_step = False

        agent_rotation = state["rotation"]
        sensor_rotation = state["sensors"][self.sensor_module_id + ".rgba"]["rotation"]
        self.state = {
            "location": agent_position + sensor_position,
            "rotation": agent_rotation * sensor_rotation,
        }

    def state_dict(self):
        state_dict = self._snapshot_telemetry.state_dict()
        state_dict.update(processed_observations=self.processed_obs)
        return state_dict

    def step(self, data) -> State | None:
        """Turn raw observations into dict of features at location.

        Args:
            data: Raw observations.

        Returns:
            State with features and morphological features. Noise may be added.
            use_state flag may be set.
        """
        if self.save_raw_obs and not self.is_exploring:
            self._snapshot_telemetry.raw_observation(
                data,
                self.state["rotation"],
                self.state["location"]
                if "location" in self.state.keys()
                else self.state["position"],
            )

        observed_state, telemetry = self._habitat_observation_processor.process(data)

        if observed_state.use_state:
            observed_state = self._message_noise(observed_state)

        if self.motor_only_step:
            # Set interesting-features flag to False, as should not be passed to
            # LM, even in e.g. pre-training experiments that might otherwise do so
            observed_state.use_state = False

        observed_state = self._state_filter(observed_state)

        if not self.is_exploring:
            self.processed_obs.append(telemetry.processed_obs.__dict__)
            self.states.append(self.state)
            self.visited_locs.append(telemetry.visited_loc)
            self.visited_normals.append(telemetry.visited_normal)

        return observed_state


class StateFilter(Protocol):
    def __call__(self, state: State) -> State: ...
    def reset(self) -> None: ...


class PassthroughStateFilter(StateFilter):
    def __call__(self, state: State) -> State:
        """Passthrough state filter. Never sets `state.use_state` to False.

        Returns:
            State unchanged.
        """
        return state

    def reset(self) -> None:
        pass


class FeatureChangeFilter(StateFilter):
    def __init__(self, delta_thresholds: dict[str, Any]):
        self._delta_thresholds = delta_thresholds
        self._last_state = None
        self._last_sent_n_steps_ago = 0

    def reset(self):
        """Reset buffer and is_exploring flag."""
        self._last_state = None

    def _check_feature_change(self, state: State) -> bool:
        """Check feature change between last transmitted observation.

        Args:
            state: State to check for feature change.

        Returns:
            True if the features have changed significantly.
        """
        if not state.get_on_object():
            # Even for the surface-agent sensor, do not return a feature for LM
            # processing that is not on the object
            logger.debug("No new point because not on object")
            return False

        for feature in self._delta_thresholds:
            if feature not in ["n_steps", "distance"]:
                last_feat = self._last_state.get_feature_by_name(feature)
                current_feat = state.get_feature_by_name(feature)

            if feature == "n_steps":
                if self._last_sent_n_steps_ago >= self._delta_thresholds[feature]:
                    logger.debug(f"new point because of {feature}")
                    return True
            elif feature == "distance":
                distance = np.linalg.norm(
                    np.array(self._last_state.location) - np.array(state.location)
                )

                if distance > self._delta_thresholds[feature]:
                    logger.debug(f"new point because of {feature}")
                    return True

            elif feature == "hsv":
                last_hue = last_feat[0]
                current_hue = current_feat[0]
                hue_d = min(
                    abs(current_hue - last_hue), 1 - abs(current_hue - last_hue)
                )
                if hue_d > self._delta_thresholds[feature][0]:
                    return True
                delta_change_sv = np.abs(last_feat[1:] - current_feat[1:])
                for i, dc in enumerate(delta_change_sv):
                    if dc > self._delta_thresholds[feature][i + 1]:
                        logger.debug(f"new point because of {feature} - {i + 1}")
                        return True

            elif feature == "pose_vectors":
                angle_between = get_angle(
                    last_feat[0],
                    current_feat[0],
                )
                if angle_between >= self._delta_thresholds[feature][0]:
                    logger.debug(
                        f"new point because of {feature} angle : {angle_between}"
                    )
                    return True

            else:
                delta_change = np.abs(last_feat - current_feat)
                if len(delta_change.shape) > 0:
                    for i, dc in enumerate(delta_change):
                        if dc > self._delta_thresholds[feature][i]:
                            logger.debug(f"new point because of {feature} - {dc}")
                            return True
                elif delta_change > self._delta_thresholds[feature]:
                    logger.debug(f"new point because of {feature}")
                    return True
        return False

    def __call__(self, state: State) -> State:
        """Sets `state.use_state` to False if features haven't changed significantly.

        Args:
            state: State to check for feature change.

        Returns:
            State with `state.use_state` set to False if features haven't changed
            significantly.
        """
        if not state.use_state:
            # If we already know the features are uninteresting (e.g. invalid surface
            # normal due to <3/4 of the object in view, or motor only-step), then
            # don't bother with the below
            return state

        if not self._last_state:  # first step
            self._last_state = state
            self._last_sent_n_steps_ago = 0
            return state

        significant_feature_change = self._check_feature_change(state)

        # Save bool which will tell us whether to pass the information to LMs
        state.use_state = significant_feature_change

        if significant_feature_change:
            # As per original implementation : only update the "last feature" when a
            # significant change has taken place
            self._last_state = state
            self._last_sent_n_steps_ago = 0
        else:
            self._last_sent_n_steps_ago += 1

        return state
