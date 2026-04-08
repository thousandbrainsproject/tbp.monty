# Copyright 2025-2026 Thousand Brains Project
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
from enum import Enum
from typing import Any, ClassVar, Protocol

import numpy as np
import quaternion as qt
from scipy.spatial.transform import Rotation
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import local_binary_pattern

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
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.frameworks.utils.sensor_processing import (
    log_sign,
    principal_curvatures,
    scale_clip,
    surface_normal_naive,
    surface_normal_ordinary_least_squares,
    surface_normal_total_least_squares,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import get_angle

__all__ = [
    "CameraSM",
    "DefaultMessageNoise",
    "FeatureChangeFilter",
    "MessageNoise",
    "NoMessageNoise",
    "ObservationProcessor",
    "PassthroughPerceptFilter",
    "PerceptFilter",
    "Probe",
    "SnapshotTelemetry",
    "SurfaceNormalMethod",
]

logger = logging.getLogger(__name__)


class SnapshotTelemetry:
    """Keeps track of raw observation snapshot telemetry."""

    def __init__(self) -> None:
        self.poses: list[dict[str, np.ndarray]] = []
        self.raw_observations: list[SensorObservation] = []

    def reset(self):
        """Reset the snapshot telemetry."""
        self.raw_observations = []
        self.poses = []

    def raw_observation(
        self,
        raw_observation: SensorObservation,
        rotation: qt.quaternion,
        position: np.ndarray,
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
                sm_rotation=qt.as_float_array(rotation),
                sm_location=np.array(position),
            )
        )

    def state_dict(
        self,
    ) -> dict[str, list[SensorObservation] | list[dict[str, np.ndarray]]]:
        """Returns recorded raw observation snapshots.

        Returns:
            Dictionary containing a list of raw observations in `raw_observations` and
            a list of pose information for each observation in `sm_properties`.
        """
        assert len(self.poses) == len(self.raw_observations), (
            "Each raw observation should have a corresponding pose information."
        )
        return dict(raw_observations=self.raw_observations, sm_properties=self.poses)


class SurfaceNormalMethod(Enum):
    TLS = "TLS"
    """Total Least-Squares"""
    OLS = "OLS"
    """Ordinary Least-Squares"""
    NAIVE = "naive"
    """Naive"""


class ObservationProcessor:
    """Processes SensorObservations into Cortical Messages."""

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
        # "mock_obs",
        "local_binary_pattern",
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
        """Initializes the ObservationProcessor.

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

    def process(self, observation: SensorObservation) -> Message:
        """Processes observation.

        Args:
            observation: Habitat observation.

        Returns:
            A Percept.
        """
        obs_3d = observation["semantic_3d"]
        sensor_frame_data = observation["sensor_frame_data"]
        world_camera = observation["world_camera"]
        rgba_feat = observation["rgba"]
        # mock_obs = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
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
                valid_signals,
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
            valid_signals = False
            morphological_features = {}

        if "on_object" in self._features:
            morphological_features["on_object"] = float(on_object)

        # Sensor module returns features at a location in the form of a Message class.
        # use_state is a bool indicating whether the input is "interesting",
        # which indicates that it merits processing by the learning module; by default
        # it will always be True so long as the surface normal and principal curvature
        # directions were valid; certain SMs and policies used separately can also set
        # it to False under appropriate conditions

        percept = Message(
            location=np.array([x, y, z]),
            morphological_features=morphological_features,
            non_morphological_features=features,
            confidence=1.0,
            use_state=on_object and valid_signals,
            sender_id=self._sensor_module_id,
            sender_type="SM",
        )
        # This is just for logging! Do not use _ attributes for matching
        percept._semantic_id = semantic_id

        return percept

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
        and principal curvature directions are well-defined.

        Returns:
            features: The features in the patch.
            morphological_features: ?
            valid_signals: True if the surface normal and principal curvature
                directions are well-defined.
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
        if "local_binary_pattern" in self._features:
            patch = rgba_feat[:, :, :3]
            gray = rgb2gray(patch)
            gray = (gray * 255).astype(np.uint8)
            p = 8
            r = 1
            lbp = local_binary_pattern(gray, p, r, method="ror")
            n_bins = int(lbp.max() + 1)
            hist, _ = np.histogram(
                lbp.ravel(),
                bins=n_bins,
                range=(0, n_bins),
            )
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-6)
            features["local_binary_pattern"] = hist

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

        valid_signals = valid_sn and valid_pc
        if not valid_signals:
            logger.debug("Either the surface-normal or pc-directions were ill-defined")

        return features, morphological_features, valid_signals

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

    def __init__(self, sensor_module_id: str, save_raw_obs: bool):
        """Initialize the probe.

        Args:
            rng: Random number generator. Unused.
            sensor_module_id: Name of sensor module.
            save_raw_obs: Whether to save raw sensory input for logging.
        """
        super().__init__()

        self.is_exploring = False
        self.sensor_module_id = sensor_module_id
        self.state: SensorState | None = None
        self.save_raw_obs = save_raw_obs

        self._snapshot_telemetry = SnapshotTelemetry()

    def state_dict(self):
        return self._snapshot_telemetry.state_dict()

    def update_state(self, agent: AgentState):
        """Update information about the sensors location and rotation."""
        sensor = agent.sensors[SensorID(self.sensor_module_id)]
        self.state = SensorState(
            position=agent.position
            + qt.rotate_vectors(agent.rotation, sensor.position),
            rotation=agent.rotation * sensor.rotation,
        )

    def step(
        self,
        ctx: RuntimeContext,  # noqa: ARG002
        observation: SensorObservation,
        motor_only_step: bool = False,  # noqa: ARG002
    ) -> Message | None:
        if self.save_raw_obs and not self.is_exploring:
            self._snapshot_telemetry.raw_observation(
                observation, self.state.rotation, self.state.position
            )

        return None

    def pre_episode(self) -> None:
        """Reset buffer and is_exploring flag."""
        self._snapshot_telemetry.reset()
        self.is_exploring = False


class MessageNoise(Protocol):
    def __call__(self, percept: Message, rng: np.random.RandomState) -> Message: ...


class NoMessageNoise(MessageNoise):
    def __call__(self, percept: Message, rng: np.random.RandomState) -> Message:  # noqa: ARG002
        """No noise function.

        Returns:
            Percept with no noise added.
        """
        return percept


class DefaultMessageNoise(MessageNoise):
    def __init__(self, noise_params: dict[str, Any]):
        self.noise_params = noise_params

    def __call__(self, percept: Message, rng: np.random.RandomState) -> Message:
        """Add noise to features specified in noise_params.

        Noise params should have structure {"features":
                                                {"feature_keys": noise_amount, ...},
                                            "locations": noise_amount}
        noise_amount specifies the standard deviation of the gaussian noise sampled
        for real valued features. For boolean features it specifies the probability
        that the boolean flips.
        If we are dealing with normed vectors (surface_normal or curvature_directions),
        the noise is applied by rotating the vector given a sampled rotation. Otherwise
        noise is just added onto the perceived feature value.

        Args:
            percept: Percept to add noise to.
            rng: Random number generator.

        Returns:
            Percept with noise added.
        """
        if "features" in self.noise_params:
            for key in self.noise_params["features"]:
                if key in percept.morphological_features:
                    if key == "pose_vectors":
                        # apply randomly sampled rotation to xyz axes with standard
                        # deviation specified in noise_params
                        # TODO: apply same rotation to both to make sure they stay
                        # orthogonal?
                        noise_angles = rng.normal(
                            0, self.noise_params["features"][key], 3
                        )
                        noise_rotation = Rotation.from_euler(
                            "xyz", noise_angles, degrees=True
                        )
                        percept.morphological_features[key] = noise_rotation.apply(
                            percept.morphological_features[key]
                        )
                    else:
                        percept.morphological_features[key] = (
                            self.add_noise_to_feat_value(
                                feat_name=key,
                                feat_val=percept.morphological_features[key],
                                rng=rng,
                            )
                        )
                elif key in percept.non_morphological_features:
                    percept.non_morphological_features[key] = (
                        self.add_noise_to_feat_value(
                            feat_name=key,
                            feat_val=percept.non_morphological_features[key],
                            rng=rng,
                        )
                    )
        if "location" in self.noise_params:
            noise = rng.normal(0, self.noise_params["location"], 3)
            percept.location = percept.location + noise

        return percept

    def add_noise_to_feat_value(self, feat_name, feat_val, rng: np.random.RandomState):
        if isinstance(feat_val, bool):
            # Flip boolean variable with probability specified in
            # noise_params
            if rng.random() < self.noise_params["features"][feat_name]:
                new_feat_val = not (feat_val)
            else:
                new_feat_val = feat_val

        else:
            # Add gaussian noise with standard deviation specified in
            # noise_params
            shape = feat_val.shape
            noise = rng.normal(0, self.noise_params["features"][feat_name], shape)
            new_feat_val = feat_val + noise
            if feat_name == "hsv":  # make sure hue stays in 0-1 range
                new_feat_val[0] = np.clip(new_feat_val[0], 0, 1)
        return new_feat_val


class CameraSM(SensorModule):
    """Sensor Module that turns RGBD camera observations into features at locations.

    Takes in camera rgba and depth input and calculates locations from this.
    It also extracts features which are currently: on_object, rgba, surface_normal,
    curvature.
    """

    def __init__(
        self,
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
            When using feature-at-location matching with graphs, surface_normal and
            on_object need to be in the list of features.

        Note:
            gaussian_curvature and mean_curvature should be used together to preserve
            the same information contained in principal_curvatures.
        """
        self._observation_processor = ObservationProcessor(
            features=features,
            sensor_module_id=sensor_module_id,
            pc1_is_pc2_threshold=pc1_is_pc2_threshold,
            is_surface_sm=is_surface_sm,
        )
        # TODO: With DefaultMessageNoise not getting RNG on init anymore,
        #       then we can initialize CameraSM with MessageNoise, instead
        #       of noise_params.
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
        # Tests check sm.features, not sure if this should be exposed
        self.features = features
        self.processed_obs: list[dict[str, Any]] = []
        self.states: list[SensorState] = []
        # TODO: give more descriptive & distinct names
        self.sensor_module_id = sensor_module_id
        self.save_raw_obs = save_raw_obs

    def pre_episode(self) -> None:
        self._snapshot_telemetry.reset()
        self._percept_filter.reset()
        self.is_exploring = False
        self.processed_obs = []
        self.states = []

    def update_state(self, agent: AgentState):
        """Update information about the sensors location and rotation."""
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
    ) -> Message | None:
        """Turn raw observations into dict of features at location.

        Args:
            ctx: The runtime context.
            observation: Raw sensor observation.
            motor_only_step: Whether the current step is a motor-only step.

        Returns:
            Percept with features and morphological features. Noise may be
            added. The `use_state` flag may be set.
        """
        if self.save_raw_obs and not self.is_exploring:
            self._snapshot_telemetry.raw_observation(
                observation, self.state.rotation, self.state.position
            )

        percept = self._observation_processor.process(observation)

        if percept.use_state:
            percept = self._message_noise(percept, rng=ctx.rng)

        if motor_only_step:
            percept.use_state = False

        percept = self._percept_filter(percept)

        if not self.is_exploring:
            self.processed_obs.append(percept.__dict__)
            self.states.append(self.state)

        return percept


class PerceptFilter(Protocol):
    def __call__(self, percept: Message) -> Message: ...
    def reset(self) -> None: ...


class PassthroughPerceptFilter(PerceptFilter):
    def __call__(self, percept: Message) -> Message:
        """Passthrough percept filter. Never sets `percept.use_state` to False.

        Returns:
            Percept unchanged.
        """
        return percept

    def reset(self) -> None:
        pass


class FeatureChangeFilter(PerceptFilter):
    def __init__(self, delta_thresholds: dict[str, Any]):
        self._delta_thresholds = delta_thresholds
        self._last_percept = None
        self._last_sent_n_steps_ago = 0

    def reset(self):
        """Reset buffer and is_exploring flag."""
        self._last_percept = None

    def _check_feature_change(self, percept: Message) -> bool:
        """Check feature change between last transmitted observation.

        Args:
            percept: Percept to check for feature change.

        Returns:
            True if the features have changed significantly.
        """
        if not percept.get_on_object():
            # Even for the surface-agent sensor, do not return a feature for LM
            # processing that is not on the object
            logger.debug("No new point because not on object")
            return False

        for feature in self._delta_thresholds:
            if feature not in ["n_steps", "distance"]:
                last_feat = self._last_percept.get_feature_by_name(feature)
                current_feat = percept.get_feature_by_name(feature)

            if feature == "n_steps":
                if self._last_sent_n_steps_ago >= self._delta_thresholds[feature]:
                    logger.debug(f"new point because of {feature}")
                    return True
            elif feature == "distance":
                distance = np.linalg.norm(
                    np.array(self._last_percept.location) - np.array(percept.location)
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
            elif feature == "local_binary_pattern":
                pass

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

    def __call__(self, percept: Message) -> Message:
        """Sets `percept.use_state` to False if features haven't changed significantly.

        Args:
            percept: Percept to check for feature change.

        Returns:
            Percept with `percept.use_state` set to False if features haven't
            changed significantly.
        """
        if not percept.use_state:
            # If we already know the features are uninteresting (e.g. invalid surface
            # normal due to <3/4 of the object in view, or motor only-step), then
            # don't bother with the below
            return percept

        if not self._last_percept:  # first step
            self._last_percept = percept  # type: ignore[assignment]
            self._last_sent_n_steps_ago = 0
            return percept

        significant_feature_change = self._check_feature_change(percept)

        # Save bool which will tell us whether to pass the information to LMs
        percept.use_state = significant_feature_change

        if significant_feature_change:
            # As per original implementation : only update the "last feature" when a
            # significant change has taken place
            self._last_percept = percept
            self._last_sent_n_steps_ago = 0
        else:
            self._last_sent_n_steps_ago += 1

        return percept
