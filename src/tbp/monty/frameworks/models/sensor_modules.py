# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import logging

import numpy as np
import quaternion
from scipy.spatial.transform import Rotation
from skimage.color import rgb2hsv

from tbp.monty.frameworks.models.monty_base import SensorModuleBase
from tbp.monty.frameworks.models.states import State
from tbp.monty.frameworks.utils.sensor_processing import (
    get_point_normal_naive,
    get_point_normal_ordinary_least_squares,
    get_point_normal_total_least_squares,
    get_principal_curvatures,
    log_sign,
    scale_clip,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import get_angle


class DetailedLoggingSM(SensorModuleBase):
    """Sensor module that keeps track of raw observations for logging."""

    def __init__(
        self,
        sensor_module_id,
        save_raw_obs,
        pc1_is_pc2_threshold=10,
        point_normal_method="TLS",
        weight_curvature=True,
    ):
        """Initialize Sensor Module.

        Args:
            sensor_module_id: Name of sensor module.
            save_raw_obs: Whether to save raw sensory input for logging.
            pc1_is_pc2_threshold: maximum difference between pc1 and pc2 to be
                classified as being roughly the same (ignore curvature directions).
            point_normal_method: in ['TLS' (default), 'OLS', 'naive']. Determines which
                implementation to use for point-normal extraction ("TLS" stands for
                total least-squares (default), "OLS" for ordinary least-squares, 'naive'
                for the original tangent vector cross-product implementation). Any other
                value will raise an error.
            weight_curvature: determines whether to use the "weighted" (True) or
                "unweighted" (False) implementation for principal curvature extraction.
        """
        super(DetailedLoggingSM, self).__init__(sensor_module_id)
        self.save_raw_obs = save_raw_obs
        self.raw_observations = []
        self.sm_properties = []
        self.pc1_is_pc2_threshold = pc1_is_pc2_threshold
        self.point_normal_method = point_normal_method
        self.weight_curvature = weight_curvature

    def state_dict(self):
        """Return state_dict."""
        # this is what is saved to detailed stats
        assert len(self.sm_properties) == len(
            self.raw_observations
        ), "Should have a SM value for every set of observations."

        return dict(
            raw_observations=self.raw_observations, sm_properties=self.sm_properties
        )

    def step(self, data):
        """Add raw observations to SM buffer."""
        if self.save_raw_obs and not self.is_exploring:
            self.raw_observations.append(data)
            # save the sensor state at every step

            if self.state is not None:
                # "position" key available for DetailedLoggingSM, "location" key
                # for e.g. HabitatDistantPatchSM, which accounts for both agent
                # and sensory positions; TODO consider making these keys
                # more consistent
                if "position" in self.state.keys():
                    self.sm_properties.append(
                        dict(
                            sm_rotation=quaternion.as_float_array(
                                self.state["rotation"]
                            ),
                            sm_location=np.array(self.state["position"]),
                        )
                    )
                elif "location" in self.state.keys():
                    self.sm_properties.append(
                        dict(
                            sm_rotation=quaternion.as_float_array(
                                self.state["rotation"]
                            ),
                            sm_location=np.array(self.state["location"]),
                        )
                    )

    def pre_episode(self):
        """Reset buffer and is_exploring flag."""
        self.raw_observations = []
        self.sm_properties = []
        self.is_exploring = False

        # Store visited locations in global environment coordinates to help inform
        # more intelligent motor-policies
        # TODO consider adding a flag or mixin to determine when these are actually
        # saved
        self.visited_locs = []
        self.visited_normals = []

    def extract_and_add_features(
        self,
        features,
        obs_3d,
        rgba_feat,
        depth_feat,
        center_id,
        center_row_col,
        sensor_frame_data,
        world_camera,
    ):
        """Extract features specified in self.features from sensor patch.

        Returns the features in the patch, and True if the point-normal
        or principal curvature directions were ill-defined.

        Returns:
            features: The features in the patch.
            morphological_features: ?
            invalid_signals: True if the point-normal or principal curvature directions
                were ill-defined.
        """
        # ------------ Extract Morphological Features ------------
        # Get point normal for graph matching with features
        point_normal, valid_pn = self._get_point_normals(
            obs_3d, sensor_frame_data, center_id, world_camera
        )

        k1, k2, dir1, dir2, valid_pc = get_principal_curvatures(
            obs_3d, center_id, point_normal, weighted=self.weight_curvature
        )
        # TODO: test using log curvatures instead
        if np.abs(k1 - k2) < self.pc1_is_pc2_threshold:
            pose_fully_defined = False
        else:
            pose_fully_defined = True

        morphological_features = {
            "pose_vectors": np.vstack(
                [
                    point_normal,
                    dir1,
                    dir2,
                ]
            ),
            "pose_fully_defined": pose_fully_defined,
        }
        # ---------- Extract Optional, Non-Morphological Features ----------
        if "rgba" in self.features:
            features["rgba"] = rgba_feat[center_row_col, center_row_col]
        if "min_depth" in self.features:
            features["min_depth"] = np.min(depth_feat[obs_3d[:, 3] != 0])
        if "mean_depth" in self.features:
            features["mean_depth"] = np.mean(depth_feat[obs_3d[:, 3] != 0])
        if "hsv" in self.features:
            rgba = rgba_feat[center_row_col, center_row_col]
            hsv = rgb2hsv(rgba[:3])
            features["hsv"] = hsv

        # Note we only determine curvature if we could determine a valid point-normal
        if any("curvature" in feat for feat in self.features) and valid_pn:
            if valid_pc:
                # Only process the below features if the principal curvature was valid,
                # and therefore we have a defined k1, k2 etc.
                if "principal_curvatures" in self.features:
                    features["principal_curvatures"] = np.array([k1, k2])

                if "principal_curvatures_log" in self.features:
                    features["principal_curvatures_log"] = log_sign(np.array([k1, k2]))

                if "gaussian_curvature" in self.features:
                    features["gaussian_curvature"] = k1 * k2

                if "mean_curvature" in self.features:
                    features["mean_curvature"] = (k1 + k2) / 2

                if "gaussian_curvature_sc" in self.features:
                    gc = k1 * k2
                    gc_scaled_clipped = scale_clip(gc, 4096)
                    features["gaussian_curvature_sc"] = gc_scaled_clipped

                if "mean_curvature_sc" in self.features:
                    mc = (k1 + k2) / 2
                    mc_scaled_clipped = scale_clip(mc, 256)
                    features["mean_curvature_sc"] = mc_scaled_clipped
        else:
            # Flag that PC directions are non-meaningful for e.g. downstream motor
            # policies
            features["pose_fully_defined"] = False

        invalid_signals = (not valid_pn) or (not valid_pc)
        if invalid_signals:
            logging.debug("Either the point-normal or pc-directions were ill-defined")

        return features, morphological_features, invalid_signals

    def observations_to_comunication_protocol(self, data, on_object_only=True):
        """Turn raw observations into instance of State class following CMP.

        Args:
            data: Raw observations.
            on_object_only: If False, do the following:
                - If the center of the image is not on the object, but some other part
                    of the object is in the image, continue with feature extraction
                - Get the point normal for the whole image, not just the parts of the
                    image that include an object.

        Returns:
            State: Features and morphological features.
        """
        obs_3d = data["semantic_3d"]
        sensor_frame_data = data["sensor_frame_data"]
        world_camera = data["world_camera"]
        rgba_feat = data["rgba"]
        depth_feat = data["depth"].reshape(data["depth"].size, 1).astype(np.float64)
        # Assuming squared patches
        center_row_col = rgba_feat.shape[0] // 2
        # Calculate center ID for flat semantic obs
        obs_dim = int(np.sqrt(obs_3d.shape[0]))
        half_obs_dim = obs_dim // 2
        center_id = half_obs_dim + obs_dim * half_obs_dim
        # Extract all specified features
        features = dict()
        if "object_coverage" in self.features:
            # Last dimension is semantic ID (integer >0 if on any object)
            features["object_coverage"] = sum(obs_3d[:, 3] > 0) / len(obs_3d[:, 3])
            assert (
                features["object_coverage"] <= 1.0
            ), "Coverage cannot be greater than 100%"

        if obs_3d[center_id][3] or (
            not on_object_only and features["object_coverage"] > 0
        ):
            (
                features,
                morphological_features,
                invalid_signals,
            ) = self.extract_and_add_features(
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
            morphological_features = dict()

        obs_3d_center = obs_3d[center_id]
        x, y, z, semantic_id = obs_3d_center
        if "on_object" in self.features:
            morphological_features["on_object"] = float(semantic_id > 0)

        # Sensor module returns features at a location in the form of a State class.
        # use_state is a bool indicating whether the input is "interesting",
        # which indicates that it merits processing by the learning module; by default
        # it will always be True so long as the point-normal and principal curvature
        # directions were valid; certain SMs and policies used separately can also set
        # it to False under appropriate conditions

        observed_state = State(
            location=np.array([x, y, z]),
            morphological_features=morphological_features,
            non_morphological_features=features,
            confidence=1.0,
            use_state=bool(morphological_features["on_object"]) and not invalid_signals,
            sender_id=self.sensor_module_id,
            sender_type="SM",
        )
        # This is just for logging! Do not use _ attributes for matching
        observed_state._semantic_id = semantic_id

        # Save raw observations and state for logging, and for use by
        # specialized motor-policies
        if not self.is_exploring:
            # TODO: only if using detailed logger?
            self.processed_obs.append(observed_state.__dict__)
            self.states.append(self.state)

            self.visited_locs.append(observed_state.location)

            if "pose_vectors" in morphological_features.keys():
                self.visited_normals.append(morphological_features["pose_vectors"][0])
            else:
                self.visited_normals.append(None)

        return observed_state

    def _get_point_normals(self, obs_3d, sensor_frame_data, center_id, world_camera):
        if self.point_normal_method == "TLS":
            # Version with Total Least-Squares (TLS) fitting
            point_normal, valid_pn = get_point_normal_total_least_squares(
                obs_3d, center_id, world_camera[:3, 2]
            )
        elif self.point_normal_method == "OLS":
            # Version with Ordinary Least-Squares (TLS) fitting
            point_normal, valid_pn = get_point_normal_ordinary_least_squares(
                sensor_frame_data, world_camera, center_id
            )
        elif self.point_normal_method == "naive":
            # Naive version
            point_normal, valid_pn = get_point_normal_naive(
                obs_3d, patch_radius_frac=2.5
            )
        # old version for estimating with open3d (slow on lambda node)
        # to use, uncomment lines below and import open3d
        # elif self.point_normal_method == "open3d":
        #     point_normal_alt = get_point_normal_open3d(
        #         obs_3d,
        #         center_id,
        #         sensor_location=self.state["location"],
        #         on_object_only=True,
        #     )
        else:
            raise ValueError(
                "point_normal_method must be in ['TLS' (default), 'OLS', 'naive']."
            )

        return point_normal, valid_pn


class NoiseMixin:
    def __init__(self, noise_params, **kwargs):
        super().__init__(**kwargs)
        self.noise_params = noise_params

    def add_noise_to_sensor_data(self, sensor_data):
        """Add noise to features specified in noise_params.

        Noise params should have structure {"features":
                                                {"feature_keys": noise_amount, ...},
                                            "locations": noise_amount}
        noise_amount specifies the standard deviation of the gaussian noise sampled
        for real valued features. For boolian features it specifies the probability
        that the boolean flips.
        If we are dealing with normed vectors (point_normal or curvature_directions)
        the noise is applied by rotating the vector given a sampled rotation. Otherwise
        noise is just added onto the perceived feature value.

        Args:
            sensor_data: Sensor data to add noise to.

        Returns:
            Sensor data with noise added.
        """
        if "features" in self.noise_params.keys():
            for key in self.noise_params["features"].keys():
                if key in sensor_data.morphological_features.keys():
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
                        sensor_data.morphological_features[key] = noise_rotation.apply(
                            sensor_data.morphological_features[key]
                        )
                    else:
                        sensor_data.morphological_features[key] = (
                            self.add_noise_to_feat_value(
                                feat_name=key,
                                feat_val=sensor_data.morphological_features[key],
                            )
                        )
                elif key in sensor_data.non_morphological_features.keys():
                    sensor_data.non_morphological_features[key] = (
                        self.add_noise_to_feat_value(
                            feat_name=key,
                            feat_val=sensor_data.non_morphological_features[key],
                        )
                    )
        if "location" in self.noise_params.keys():
            noise = self.rng.normal(0, self.noise_params["location"], 3)
            sensor_data.location = sensor_data.location + noise

        return sensor_data

    def add_noise_to_feat_value(self, feat_name, feat_val):
        if type(feat_val) == bool:
            # Flip boolian variable with probability specified in
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


class HabitatDistantPatchSM(DetailedLoggingSM, NoiseMixin):
    """Sensor Module that turns Habitat camera obs into features at locations.

    Takes in camera rgba and depth input and calculates locations from this.
    It also extracts features which are currently: on_object, rgba, point_normal,
    curvature.
    """

    def __init__(
        self,
        sensor_module_id,
        features,
        save_raw_obs=False,
        pc1_is_pc2_threshold=10,
        noise_params=None,
        process_all_obs=False,
    ):
        """Initialize Sensor Module.

        Args:
            sensor_module_id: Name of sensor module.
            features: Which features to extract. In [on_object, rgba, point_normal,
                principal_curvatures, curvature_directions, gaussian_curvature,
                mean_curvature]
            save_raw_obs: Whether to save raw sensory input for logging.
            pc1_is_pc2_threshold: ?. Defaults to 10.
            noise_params: Dictionary of noise amount for each feature.
            process_all_obs: Enable explicitly to enforce that off-observations are
                still processed by LMs, primarily for the purpose of unit testing.
                TODO: remove?

        Note:
            When using feature at location matching with graphs, point_normal and
            on_object needs to be in the list of features.

        Note:
            gaussian_curvature and mean_curvature should be used together to contain
            the same information as principal_curvatures.
        """
        super(HabitatDistantPatchSM, self).__init__(
            sensor_module_id, save_raw_obs, pc1_is_pc2_threshold
        )
        NoiseMixin.__init__(self, noise_params)
        possible_features = [
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
        for feature in features:
            assert (
                feature in possible_features
            ), f"{feature} not part of {possible_features}"

        self.features = features
        self.processed_obs = []
        self.states = []
        # TODO: give more descriptive & distinct names
        self.on_object_obs_only = True
        self.process_all_obs = process_all_obs

    def pre_episode(self):
        """Reset buffer and is_exploring flag."""
        super().pre_episode()
        self.processed_obs = []
        self.states = []

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
        """Return state_dict."""
        assert len(self.sm_properties) == len(
            self.raw_observations
        ), "Should have a SM value for every set of observations."

        return dict(
            raw_observations=self.raw_observations,
            processed_observations=self.processed_obs,
            sm_properties=self.sm_properties,
            # sensor_states=self.states, # pickle problem with magnum
        )

    def step(self, data):
        """Turn raw observations into dict of features at location.

        Args:
            data: Raw observations.

        Returns:
            State with features and morphological features. Noise may be added.
            use_state flag may be set.
        """
        super().step(data)  # for logging
        observed_state = self.observations_to_comunication_protocol(
            data, on_object_only=self.on_object_obs_only
        )

        if self.noise_params is not None and observed_state.use_state:
            observed_state = self.add_noise_to_sensor_data(observed_state)
        if self.process_all_obs:
            observed_state.use_state = True

        if self.motor_only_step:
            # Set interesting-features flag to False, as should not be passed to
            # LM, even in e.g. pre-training experiments that might otherwise do so
            observed_state.use_state = False

        return observed_state


class HabitatSurfacePatchSM(HabitatDistantPatchSM):
    """HabitatDistantPatchSM that continues feature extraction when patch not on object.

    Identical to HabitatDistantPatchSM except that feature extraction continues even
    if the center of the sensor patch is not on the object.
    TODO: remove and replace with surf_agent_sm=True.
    """

    def __init__(
        self, sensor_module_id, features, save_raw_obs=False, noise_params=None
    ):
        super().__init__(
            sensor_module_id, features, save_raw_obs, noise_params=noise_params
        )

        self.on_object_obs_only = False  # parameter used in step() method


class FeatureChangeSM(HabitatDistantPatchSM, NoiseMixin):
    """Sensor Module that turns Habitat camera obs into features at locations.

    Takes in camera rgba and depth input and calculates locations from this.
    It also extracts features which are currently: on_object, rgba, point_normal,
    curvature.
    """

    def __init__(
        self,
        sensor_module_id,
        features,
        delta_thresholds,
        surf_agent_sm=False,
        save_raw_obs=False,
        noise_params=None,
    ):
        """Initialize Sensor Module.

        Args:
            sensor_module_id: Name of sensor module.
            features: Which features to extract. In [on_object, rgba, point_normal,
                principal_curvatures, curvature_directions, gaussian_curvature,
                mean_curvature]
            delta_thresholds: thresholds for each feature to be considered a
                significant change.
            surf_agent_sm: Boolean that is False by default, indicating that the
                FeatureChangeSM is used for the distant-agent; if True, used to assign
                appropriate value for self.on_object_obs_only
            save_raw_obs: Whether to save raw sensory input for logging. Defaults to
                False.
            noise_params: ?. Defaults to None.
        """
        super(FeatureChangeSM, self).__init__(
            sensor_module_id, features, save_raw_obs, noise_params=noise_params
        )
        self.delta_thresholds = delta_thresholds
        self.on_object_obs_only = not (
            surf_agent_sm
        )  # If using surface-agent approach,
        # then should be False; for distant-agent SMs, it should be True
        self.last_features = None
        self.last_sent_n_steps_ago = 0

    def pre_episode(self):
        """Reset buffer and is_exploring flag."""
        super().pre_episode()
        self.last_features = None

    def step(self, data):
        """Return Features if they changed significantly."""
        patch_observation = super().step(data)  # get extracted features

        if not patch_observation.use_state:
            # If we already know the features are uninteresting (e.g. invalid point
            # normal due to <3/4 of the object in view, or motor only-step), then
            # don't bother with the below

            return patch_observation

        if self.last_features is None:  # first step
            logging.debug("Performing first sensation step of FeatureChangeSM")
            self.last_features = patch_observation
            return patch_observation

        else:
            logging.debug("Performing FeatureChangeSM step")
            significant_feature_change = self.check_feature_change(patch_observation)

            # Save bool which will tell us whether to pass the information to LMs
            patch_observation.use_state = significant_feature_change

            if significant_feature_change:
                # As per original implementation : only update the "last feature" when a
                # significant change has taken place
                self.last_features = patch_observation
                self.last_sent_n_steps_ago = 0
            else:
                self.last_sent_n_steps_ago += 1

            return patch_observation

    def check_feature_change(self, observed_features):
        """Check feature change between last transmitted observation.

        Args:
            observed_features: Features from the current observation.

        Returns:
            True if the features have changed significantly.
        """
        if not observed_features.get_on_object():
            # Even for the surface-agent sensor, do not return a feature for LM
            # processing that is not on the object
            logging.debug(f"No new point because not on object")
            return False

        for feature in self.delta_thresholds.keys():
            if feature not in ["n_steps", "distance"]:
                last_feat = self.last_features.get_feature_by_name(feature)
                current_feat = observed_features.get_feature_by_name(feature)

            if feature == "n_steps":
                if self.last_sent_n_steps_ago >= self.delta_thresholds[feature]:
                    logging.debug(f"new point because of {feature}")
                    return True
            elif feature == "distance":
                distance = np.linalg.norm(
                    np.array(self.last_features.location)
                    - np.array(observed_features.location)
                )

                if distance > self.delta_thresholds[feature]:
                    logging.debug(f"new point because of {feature}")
                    return True

            elif feature == "hsv":
                last_hue = last_feat[0]
                current_hue = current_feat[0]
                hue_d = min(
                    abs(current_hue - last_hue), 1 - abs(current_hue - last_hue)
                )
                if hue_d > self.delta_thresholds[feature][0]:
                    return True
                delta_change_sv = np.abs(last_feat[1:] - current_feat[1:])
                for i, dc in enumerate(delta_change_sv):
                    if dc > self.delta_thresholds[feature][i + 1]:
                        logging.debug(f"new point because of {feature} - {i+1}")
                        return True

            elif feature == "pose_vectors":
                angle_between = get_angle(
                    last_feat[0],
                    current_feat[0],
                )
                if angle_between >= self.delta_thresholds[feature][0]:
                    logging.debug(
                        f"new point because of {feature} angle : {angle_between}"
                    )
                    return True

            else:
                delta_change = np.abs(last_feat - current_feat)
                if len(delta_change.shape) > 0:
                    for i, dc in enumerate(delta_change):
                        if dc > self.delta_thresholds[feature][i]:
                            logging.debug(f"new point because of {feature} - {dc}")
                            return True
                else:
                    if delta_change > self.delta_thresholds[feature]:
                        logging.debug(f"new point because of {feature}")
                        return True
        return False
