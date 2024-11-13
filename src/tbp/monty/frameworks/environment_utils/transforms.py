# Copyright 2021-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import numpy as np
import quaternion as qt
import scipy

__all__ = [
    "MissingToMaxDepth",
    "AddNoiseToRawDepthImage",
    "GaussianSmoothing",
    "DepthTo3DLocations",
]


class MissingToMaxDepth:
    """Return max depth when no mesh is present at a location.

    Habitat depth sensors return 0 when no mesh is present at a location. Instead,
    return max_depth. See:
    https://github.com/facebookresearch/habitat-sim/issues/1157 for discussion.
    """

    def __init__(self, agent_id, max_depth, threshold=0):
        """Initialize the transform.

        Args:
            agent_id: agent id of the agent where the transform should be applied.
            max_depth: numeric that will replace missing
            threshold: (optional) numeric, anything less than this is counted as
                missing. Defaults to 0.
        """
        self.agent_id = agent_id
        self.max_depth = max_depth
        self.threshold = threshold
        self.needs_rng = False

    def __call__(self, observation, state=None):
        """Replace missing depth values with max_depth.

        Args:
            observation: observation to modify in place.
            state: not used.

        Returns:
            observation, same as input, with missing data modified in place
        """
        # loop over sensor modules
        for sm in observation[self.agent_id].keys():
            m = np.where(observation[self.agent_id][sm]["depth"] <= self.threshold)
            observation[self.agent_id][sm]["depth"][m] = self.max_depth
        return observation


class AddNoiseToRawDepthImage:
    """Add gaussian noise to raw sensory input."""

    def __init__(self, agent_id, sigma):
        """Initialize the transform.

        Args:
            agent_id: agent id of the agent where the transform should be applied.
                Transform will be applied to all depth sensors of the agent.
            sigma: standard deviation of noise distribution.
        """
        self.agent_id = agent_id
        self.sigma = sigma
        self.needs_rng = True

    def __call__(self, observation, state=None):
        """Add gaussian noise to raw sensory input.

        Args:
            observation: observation to modify in place.
            state: not used.

        Returns:
            observation, same as input, with added gaussian noise to depth values.

        Raises:
            Exception: if no depth sensor is present.
        """
        # loop over sensor modules
        for sm in observation[self.agent_id].keys():
            if "depth" in observation[self.agent_id][sm].keys():
                noise = self.rng.normal(
                    0,
                    self.sigma,
                    observation[self.agent_id][sm]["depth"].shape,
                )
                observation[self.agent_id][sm]["depth"] += noise
            else:
                raise Exception("NO DEPTH SENSOR PRESENT. Don't use this transform")
        return observation


class GaussianSmoothing:
    """Deals with gaussian noise on the raw depth image.

    This transform is designed to deal with gaussian noise on the raw depth
    image. It remains to be tested whether it will also help with the kind of noise
    in a real-world depth camera.
    """

    def __init__(self, agent_id, sigma=2, kernel_width=3):
        """Initialize the transform.

        Args:
            agent_id: agent id of the agent where the transform should be applied.
                Transform will be applied to all depth sensors of the agent.
            sigma: sigma of gaussian smoothing kernel. Default is 2.
            kernel_width: width of the smoothing kernel. Default is 3.
        """
        self.agent_id = agent_id
        self.sigma = sigma
        self.kernel_width = kernel_width
        self.pad_size = kernel_width // 2
        self.kernel = self.create_kernel()
        self.needs_rng = False

    def __call__(self, observation, state=None):
        """Apply gaussian smoothing to depth images.

        Args:
            observation: observation to modify in place.
            state: not used.

        Returns:
            observation, same as input, with smoothed depth values.

        Raises:
            Exception: if no depth sensor is present.
        """
        # loop over sensor modules
        for sm in observation[self.agent_id].keys():
            if "depth" in observation[self.agent_id][sm].keys():
                depth_img = observation[self.agent_id][sm]["depth"].copy()
                padded_img = self.get_padded_img(depth_img, pad_type="edge")
                filtered_img = scipy.signal.convolve(
                    padded_img, self.kernel, mode="valid"
                )
                observation[self.agent_id][sm]["depth"] = filtered_img
            else:
                raise Exception("NO DEPTH SENSOR PRESENT. Don't use this transform")
        return observation

    def create_kernel(self):
        """Create a normalized gaussian kernel.

        Returns:
            normalized gaussian kernel. Array of size (kernel_width, kernel_width).
        """
        x = np.linspace(-self.pad_size, self.pad_size, self.kernel_width)
        kernel_1d = (
            1.0
            / (np.sqrt(2 * np.pi) * self.sigma)
            * np.exp(-np.square(x) / (2 * self.sigma**2))
        )
        kernel_2d = np.outer(kernel_1d, kernel_1d)
        return kernel_2d / np.sum(kernel_2d)

    def get_padded_img(self, img, pad_type="edge"):
        if pad_type == "edge":
            padded_img = np.pad(img.astype(float), pad_width=self.pad_size, mode="edge")
        elif pad_type == "empty":
            padded_img = np.pad(
                img.astype(float),
                pad_width=self.pad_size,
                mode="constant",
                constant_values=np.nan,
            )
        return padded_img

    def conv2d(self, img, kernel_renorm=False):
        """Apply a 2D convolution to the image.

        Args:
            img: 2D image to be filtered.
            kernel_renorm: flag that specifies whether kernel values should be
                renormalized (based on the number on non-NaN values in image window).

        Returns:
            filtered version of the input image.
        """
        [n_rows, n_cols] = img.shape
        filtered_img = img[
            self.pad_size : (n_rows - self.pad_size),
            self.pad_size : (n_cols - self.pad_size),
        ].copy()
        # TODO: Investigate vectorizing this
        for i in range(n_rows - self.kernel_width + 1):
            for j in range(n_cols - self.kernel_width + 1):
                # Extracts image subset to be averaged out by the smoothing kernel.
                # Identify indices of non-NaN values, and sum the corresponding kernel
                # weights to get the normalization factor.
                img_subset = img[i : i + self.kernel_width, j : j + self.kernel_width]
                mask = ~np.isnan(img_subset)
                norm_factor = np.sum(mask * self.kernel) if kernel_renorm else 1.0
                normalized_kernel = self.kernel / norm_factor
                filtered_img[i, j] = np.nansum(normalized_kernel * img_subset)
        return filtered_img


class DepthTo3DLocations:
    """Transform semantic and depth observations from 2D into 3D.

    Transform semantic and depth observations from camera coordinate (2D) into
    agent (or world) coordinate (3D).

    This transform will add the transformed results as a new observation called
    "semantic_3d" which will contain the 3d coordinates relative to the agent
    (or world) with the semantic ID and 3D location of every object observed::

        "semantic_3d" : [
        #    x-pos      , y-pos     , z-pos      , semantic_id
            [-0.06000001, 1.56666668, -0.30000007, 25.],
            [ 0.06000001, 1.56666668, -0.30000007, 25.],
            [-0.06000001, 1.43333332, -0.30000007, 25.],
            [ 0.06000001, 1.43333332, -0.30000007, 25.]])
        ]

    Attributes:
        agent_id: Agent ID to get observations from
        resolution: Camera resolution (H, W)
        zoom: Camera zoom factor. Defaul 1.0 (no zoom)
        hfov: Camera HFOV, default 90 degrees
        semantic_sensor: Semantic sensor id. Default "semantic"
        depth_sensor: Depth sensor id. Default "depth"
        world_coord: Whether to return 3D locations in world coordinates.
            If enabled, then :meth:`__call__` must be called with
            the agent and sensor states in addition to observations.
            Default True.
        get_all_points: Whether to return all 3D coordinates or only the ones
            that land on an object.
        depth_clip_sensors: tuple of sensor indices to which to apply a clipping
            transform where all values > clip_value are set to
            clip_value. Empty tuple ~ apply to none of them.
        clip_value: depth parameter for the clipping transform

    Warning:
        This transformation is only valid for pinhole cameras
    """

    def __init__(
        self,
        agent_id,
        sensor_ids,
        resolutions,
        zooms=1.0,
        hfov=90.0,
        clip_value=0.05,
        depth_clip_sensors=(),
        world_coord=True,
        get_all_points=False,
        use_semantic_sensor=True,
    ):
        self.needs_rng = False

        self.inv_k = []
        self.h, self.w = [], []

        if isinstance(zooms, (int, float)):
            zooms = [zooms] * len(sensor_ids)

        if isinstance(hfov, (int, float)):
            hfov = [hfov] * len(sensor_ids)

        for i, zoom in enumerate(zooms):
            # Pinhole camera, focal length fx = fy
            hfov[i] = float(hfov[i] * np.pi / 180.0)

            fx = np.tan(hfov[i] / 2.0) / zoom
            fy = fx

            # Adjust fy for aspect ratio
            self.h.append(resolutions[i][0])
            self.w.append(resolutions[i][1])
            fy = fy * self.h[i] / self.w[i]

            # Intrinsic matrix, K
            # Assuming skew is 0 for pinhole camera and center at (0,0)
            k = np.array(
                [
                    [1.0 / fx, 0.0, 0.0, 0.0],
                    [0.0, 1 / fy, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            # Inverse K
            self.inv_k.append(np.linalg.inv(k))

        self.agent_id = agent_id
        self.sensor_ids = sensor_ids
        self.world_coord = world_coord
        self.get_all_points = get_all_points
        self.use_semantic_sensor = use_semantic_sensor
        self.clip_value = clip_value
        self.depth_clip_sensors = depth_clip_sensors

    def __call__(self, observations, state=None):
        for i, sensor_id in enumerate(self.sensor_ids):
            agent_obs = observations[self.agent_id][sensor_id]
            if i in self.depth_clip_sensors:
                self.clip(agent_obs)
            depth_obs = agent_obs["depth"]
            # if applying depth clip, then do not use depth for semantic info
            # because the depth surface now includes a sheet of pixels all
            # set to the clip_value, and this sheet can confuse the
            # get_semantic_from_depth function into thinking that it is the object
            if self.depth_clip_sensors and self.use_semantic_sensor:
                semantic_obs = agent_obs["semantic"]
            elif self.use_semantic_sensor:
                surface_obs = self.get_semantic_from_depth(depth_obs.copy())
                # set pixel to 1 if it is on the main surface and on the object
                semantic_obs = agent_obs["semantic"] * surface_obs
            else:
                semantic_obs = self.get_semantic_from_depth(depth_obs.copy())

            # Approximate true world coordinates
            x, y = np.meshgrid(
                np.linspace(-1, 1, self.w[i]), np.linspace(1, -1, self.h[i])
            )
            x = x.reshape(1, self.h[i], self.w[i])
            y = y.reshape(1, self.h[i], self.w[i])

            # Unproject 2D camera coordinates into 3D coordinates relative to the agent
            depth = depth_obs.reshape(1, self.h[i], self.w[i])
            xyz = np.vstack((x * depth, y * depth, -depth, np.ones(depth.shape)))
            xyz = xyz.reshape(4, -1)
            xyz = np.matmul(self.inv_k[i], xyz)
            sensor_frame_data = xyz.T.copy()

            if self.world_coord and state is not None:
                # Get agent and sensor states from state dictionary
                agent_state = state[self.agent_id]
                depth_state = agent_state["sensors"][sensor_id + ".depth"]
                agent_rotation = agent_state["rotation"]
                agent_rotation_matrix = qt.as_rotation_matrix(agent_rotation)
                agent_position = agent_state["position"]
                sensor_rotation = depth_state["rotation"]
                sensor_position = depth_state["position"]
                # --- Apply camera transformations to get world coordinates ---
                # Combine body and sensor rotation (since sensor rotation is relative to
                # the agent this will give us the sensor rotation in world coordinates)
                sensor_rotation_rel_world = agent_rotation * sensor_rotation
                # Calculate sensor position in world coordinates -> sensor_position is
                # in the agent's coordinate frame, so we need to rotate it first by
                # agent_rotation_matrix and then add it to the agent's position
                rotated_sensor_position = agent_rotation_matrix @ sensor_position
                sensor_translation_rel_world = agent_position + rotated_sensor_position
                # Apply the rotation and translation to get the world coordinates
                rotation_matrix = qt.as_rotation_matrix(sensor_rotation_rel_world)
                world_camera = np.eye(4)
                world_camera[0:3, 0:3] = rotation_matrix
                world_camera[0:3, 3] = sensor_translation_rel_world
                xyz = np.matmul(world_camera, xyz)

                # Add sensor-to-world coordinate frame transform, used for point-normal
                # extraction. View direction is the third column of the matrix.
                observations[self.agent_id][sensor_id]["world_camera"] = world_camera

            # Extract 3D coordinates of detected objects (semantic_id != 0)
            semantic = semantic_obs.reshape(1, -1)
            if self.get_all_points:
                semantic_3d = xyz.transpose(1, 0)
                semantic_3d[:, 3] = semantic[0]
                sensor_frame_data[:, 3] = semantic[0]

                # Add point-cloud data expressed in sensor coordinate frame. Used for
                # point-normal extraction
                observations[self.agent_id][sensor_id]["sensor_frame_data"] = (
                    sensor_frame_data
                )
            else:
                detected = semantic.any(axis=0)
                xyz = xyz.transpose(1, 0)
                semantic_3d = xyz[detected]
                semantic_3d[:, 3] = semantic[0, detected]

            # Add transformed observation to existing dict. We don't need to create
            # a deepcopy because we are appending a new observation
            observations[self.agent_id][sensor_id]["semantic_3d"] = semantic_3d
        return observations

    def clip(self, agent_obs):
        """Clip the depth and semantic data that lie beyond a certain depth threshold.

        Set the values of 0 (infinite depth) to the clip value.
        """
        if "semantic" in agent_obs.keys():
            agent_obs["semantic"][agent_obs["depth"] > self.clip_value] = 0
        agent_obs["depth"][agent_obs["depth"] > self.clip_value] = self.clip_value
        agent_obs["depth"][agent_obs["depth"] == 0] = self.clip_value

    def get_on_surface_th(self, depth_patch, min_depth_range):
        """Return a depth threshold if we have a bimodal depth distribution.

        If the depth values are in a large enough range (> min_depth_range) we may
        be looking at more than one surface within our patch. This could either be
        two disjoint surfaces of the object or the object and the background.

        To figure out if we have two disjoint sets of depth values we look at the
        histogram and check for empty bins in the middle. The center of the empty
        part if the histogram will be defined as the threshold.

        Next, we want to check if we should use the depth values above or below the
        threshold. Currently this is done by looking which side of the distribution
        is larger (occupies more space in the patch). Alternatively we could check
        which side the depth at the center of the patch is on. I'm not sure what would
        be better.

        Lastly, if we do decide to use the depth points that are further away, we need
        to make sure they are not the points that are off the object. Currently this is
        just done with a simple heuristic (depth difference < 0.1) but in the future we
        will probably have to find a better solution for this.

        Args:
            depth_patch: sensor patch observations of depth
            min_depth_range: minimum range of depth values to even be considered

        Returns:
            threshold and whether we want to use values above or below threshold
        """
        depths = np.array(depth_patch).flatten()
        flip_sign = False
        th = 1000  # just high value
        if (max(depths) - min(depths)) > min_depth_range:
            # only check for bimodal distribution if we have a large enough
            # range in depth values
            height, bins = np.histogram(
                np.array(depth_patch).flatten(), bins=8, density=False
            )
            gap = np.where(height == 0)[0]
            if len(gap) > 0:
                # There is a bimodal distribution
                gap_center = len(gap) // 2
                th_id = gap[gap_center]
                th = bins[th_id]
                # Check which side of the distribution we should use
                if np.sum(height[:th_id]) < np.sum(height[th_id:]):
                    # more points in the patch are on the further away surface
                    if (bins[-1] - bins[0]) < 0.1:
                        # not too large distance between depth values -> avoid
                        # flipping sign when off object
                        flip_sign = True
        return th, flip_sign

    def get_semantic_from_depth(self, depth_patch):
        """Return semantic patch information from heuristics on depth patch.

        Args:
            depth_patch: sensor patch observations of depth

        Returns:
            sensor patch shaped info about whether each pixel is on surface of not
        """
        # avoid large range when seeing the table (goes up to almost 100 and then
        # just using 8 bins will not work anymore)
        depth_patch[depth_patch > 1] = 1.0
        th, flip_sign = self.get_on_surface_th(depth_patch, min_depth_range=0.01)
        if flip_sign is False:
            semantic_patch = depth_patch < th
        else:
            semantic_patch = depth_patch > th
        return semantic_patch
