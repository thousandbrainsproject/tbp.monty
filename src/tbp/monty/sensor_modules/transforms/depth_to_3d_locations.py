# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import quaternion as qt
from typing_extensions import Self

from tbp.monty.sensor_modules.sensor_module import (
    Payload,
    Transform,
    TransformContext,
)


class DepthTo3DLocations(Transform):
    """Transform semantic and depth observations from 2D into 3D.

    Transform semantic and depth observations from camera coordinate (2D) into
    agent (or world) coordinate (3D).

    This transform will add the transformed results as a new observation called
    "semantic_3d" which will contain the 3D coordinates relative to the agent
    (or world) with the semantic ID and 3D location of every object observed::

        "semantic_3d" : [
        #    x-pos      , y-pos     , z-pos      , semantic_id
            [-0.06000001, 1.56666668, -0.30000007, 25.],
            [ 0.06000001, 1.56666668, -0.30000007, 25.],
            [-0.06000001, 1.43333332, -0.30000007, 25.],
            [ 0.06000001, 1.43333332, -0.30000007, 25.]])
        ]

    Attributes:
        resolution: Camera resolution (H, W).
        zoom: Camera zoom factor. Defaults to 1.0 (no zoom).
        hfov: Camera HFOV. Defaults to 90 degrees.
        clip_value: Depth parameter for the clipping transform.
        should_clip_depth: Whether to clip the depth data.
        world_coord: Whether to return 3D locations in world coordinates.
            If enabled, then :meth:`__call__` must be called with
            the agent and sensor states in addition to observations.
            Defaults to True.
        get_all_points: Whether to return all 3D coordinates or only the ones
            that land on an object.
        use_semantic_sensor: Whether to use the semantic sensor.

    Warning:
        This transformation is only valid for pinhole cameras.
    """

    _clip_value: float
    _get_all_points: bool
    _h: int
    _inv_k: npt.NDArray[np.float64]
    _should_clip_depth: bool
    _use_semantic_sensor: bool
    _w: int
    _world_coord: bool

    def __init__(
        self: Self,
        resolution: tuple[int, int],
        zoom: float = 1.0,
        hfov: float = 90.0,
        clip_value: float = 0.05,
        should_clip_depth: bool = False,
        world_coord: bool = True,
        get_all_points: bool = False,
        use_semantic_sensor: bool = False,
    ):
        # Pinhole camera, focal length fx = fy
        hfov = float(hfov * np.pi / 180.0)

        fx = np.tan(hfov / 2.0) / zoom
        fy = fx

        # Adjust fy for aspect ratio
        self._h = resolution[0]
        self._w = resolution[1]
        fy = fy * self._h / self._w

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
        self._inv_k = np.linalg.inv(k)

        self._world_coord = world_coord
        self._get_all_points = get_all_points
        self._use_semantic_sensor = use_semantic_sensor
        self._clip_value = clip_value
        self._should_clip_depth = should_clip_depth

    def __call__(
        self: Self,
        ctx: TransformContext,
        payload: Payload,
    ) -> Payload:
        """Apply the depth-to-3D-locations transform to the sensor observation.

        Applies spatial transforms to the observation and generates a mask used
        to determine which pixels are the object's surface. This modifies the
        observation in-place. Some items may be modified, such as depth values,
        while other values are added (see Returns section).

        In the first part of this function, we build a mask that indicates
        not only which part of the image is on-object, but also which part of
        the image is "on-surface". The distinction between on-object and
        on-surface arises from the fact that the field of view may contain parts
        of the object that are far away from each other. For example, we may
        be looking at the front lip of a mug, but the back lip of the mug is
        also in the field of view. When we compute surface normals or
        surface curvature for the front lip of the mug, we don't want to include
        pixels from the back lip of the mug.

        To do this, we first need a semantic map that indicates only whether
        the pixel is on- or off-object. We then analyze the depth data to
        determine whether the pixels in the field of view appear to belong to the
        same part of the object (see `get_surface_from_depth` for details). The
        intersection of these two maps forms the on-surface mask (called
        `semantic_obs`) that is embedded into the observation dict and is used
        later when performing surface normal and curvature estimation.

        How we decide to build these masks is dependent on several factors,
        such as whether we are using a distant agent or a surface agent, and
        whether a ground-truth semantic map should be used. This necessitates
        a few different code paths. Here is a brief outline of the parameters
        that reflect these factors as they are commonly used in Monty:
         - when using a surface agent, self._should_clip_depth is True.
         - Surface agents also have their depth and semantic data clipped to
           a very short range from the sensor. This is done to better match the
           short reach of a finger.
         - `self._use_semantic_sensor` is currently only used with multi-object
           experiments, and when this is `True`, the observation dict will have
           an item called "semantic". In the future, we would like to include
           semantic estimation for multi-object experiments. But until then, we
           continue to support use of the semantic sensor for multi-object
           experiments.
         - When two parts of an object are visible, we separate the two parts
           according to depth. However, the threshold we use to separate the two
           parts is dependent on distant vs surface agents. The calculated parameter
           `default_on_surface_th` changes based on whether we are using a distant
           or surface agent.

        After the mask is generated, we unproject the 2D camera coordinates into
        3D coordinates relative to the agent. We then add the transformed observation
        to the original SensorObservation.

        Args:
            ctx: The transform context.
            payload: The payload to transform. Must contain an observation with "depth"
                modality.

        Returns:
            Payload with the SensorObservation with the following possibly added:
                - "semantic_3d": 3D coordinates for each pixel. If `self._world_coord`
                    is `True` (default), then the coordinates are in the world's
                    reference frame and are in the sensor's reference frame otherwise.
                    It is structured as a 2D array with shape (n_pixels, 4) with
                    columns containing x-, y-, z-coordinates, and a semantic ID.
                - "cam_to_world": Sensor-to-world coordinate frame transform. Included
                    only when `self._world_coord` is `True` (default).
                - "sensor_frame_data": 3D coordinates for each pixel relative to the
                    sensor. Has the same structure as "semantic_3d". Included only
                    when `self._get_all_points` is `True`.
        """
        depth_patch = payload.observation["depth"]

        # We need a semantic map that masks off-object pixels. We can use the
        # ground-truth semantic map if it's available. Otherwise, we generate one
        # from the depth map and (temporarily) add it to the observation dict.
        semantic_patch: npt.NDArray[np.int_]
        if "semantic" in payload.observation:
            semantic_patch = payload.observation["semantic"]
        else:
            # The generated map uses depth observations to determine whether
            # pixels are on object using 1 meter as a threshold since
            # `MissingToMaxDepth` sets the background void to 1.
            semantic_patch = np.ones_like(depth_patch, dtype=int)
            semantic_patch[depth_patch >= 1] = 0

        # Apply depth clipping to the surface agent, and initialize the
        # surface-separation threshold for later use.
        if self._should_clip_depth:
            # Surface agent: clip depth and semantic data (in place), and set
            # the default surface-separation threshold to be very short.
            self._clip(depth_patch, semantic_patch)
            default_on_surface_th = self._clip_value
        else:
            # Distance agent: do not clip depth or semantic data, and set the
            # default surface-separation threshold to be very far away.
            default_on_surface_th = 1000.0

        # Build a mask that only includes the pixels that are on-surface, where
        # on-surface means the pixels are on-object and locally connected to
        # the center of the patch's field of view. However, if we are using a
        # surface agent and are using the semantic sensor, we may use the
        # (clipped) ground-truth semantic mask as a shortcut (though it doesn't
        # use surface estimation--just on-objectness).
        if self._should_clip_depth and self._use_semantic_sensor:
            # NOTE: this particular combination of self._should_clip_depth and
            # self._use_semantic_sensor is not commonly used at present, if ever.
            # self._should_clip_depth implies a surface agent, and
            # self._use_semantic_sensor implies multi-object experiments.
            surface_patch = payload.observation["semantic"]
        else:
            surface_patch = self._get_surface_from_depth(
                depth_patch,
                semantic_patch,
                default_on_surface_th,
            )

        # Approximate true world coordinates
        x, y = np.meshgrid(np.linspace(-1, 1, self._w), np.linspace(1, -1, self._h))
        x = x.reshape(1, self._h, self._w)
        y = y.reshape(1, self._h, self._w)

        # Unproject 2D camera coordinates into 3D coordinates relative to the agent
        depth = depth_patch.reshape(1, self._h, self._w)
        xyz = np.vstack((x * depth, y * depth, -depth, np.ones(depth.shape)))
        xyz = xyz.reshape(4, -1)
        xyz = np.matmul(self._inv_k, xyz)
        sensor_frame_data = xyz.T.copy()

        if self._world_coord:
            agent_state = ctx.agent_state
            depth_state = ctx.sensor_state
            agent_rotation = agent_state.rotation
            agent_rotation_matrix = qt.as_rotation_matrix(agent_rotation)
            agent_position = agent_state.position
            sensor_rotation = depth_state.rotation
            sensor_position = depth_state.position
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
            cam_to_world = np.identity(4)
            cam_to_world[0:3, 0:3] = rotation_matrix
            cam_to_world[0:3, 3] = sensor_translation_rel_world
            xyz = cam_to_world @ xyz

            # Add sensor-to-world coordinate frame transform, used for surface
            # normal extraction. View direction is the third column of the matrix.
            payload.observation["cam_to_world"] = cam_to_world

        # Extract 3D coordinates of detected objects (semantic_id != 0)
        semantic = surface_patch.reshape(1, -1)
        if self._get_all_points:
            semantic_3d = xyz.transpose(1, 0)
            semantic_3d[:, 3] = semantic[0]
            sensor_frame_data[:, 3] = semantic[0]

            # Add point-cloud data expressed in sensor coordinate frame. Used for
            # surface normal extraction
            payload.observation["sensor_frame_data"] = sensor_frame_data
        else:
            detected = semantic.any(axis=0)
            xyz = xyz.transpose(1, 0)
            semantic_3d = xyz[detected]
            semantic_3d[:, 3] = semantic[0, detected]

        # Add transformed observation to existing dict. We don't need to create
        # a deepcopy because we are appending a new observation
        payload.observation["semantic_3d"] = semantic_3d

        return payload

    def _clip(
        self,
        depth_patch: npt.NDArray[np.float64],
        semantic_patch: npt.NDArray[np.int_],
    ) -> None:
        """Clip the depth and semantic data that lie beyond a certain depth threshold.

        This is currently used for surface agent observations to limit the "reach"
        of the agent. Pixels beyond the clip value are set to 0 in the semantic patch,
        and the depth values beyond the clip value (or equal to 0) are set to the clip
        value.

        This function modifies `depth_patch` and `semantic_patch` in-place.

        Args:
            depth_patch: depth observations
            semantic_patch: binary mask indicating on-object locations
        """
        semantic_patch[depth_patch >= self._clip_value] = 0
        depth_patch[depth_patch > self._clip_value] = self._clip_value
        depth_patch[depth_patch == 0] = self._clip_value

    def _get_on_surface_th(
        self,
        depth_patch: npt.NDArray[np.float64],
        semantic_patch: npt.NDArray[np.int_],
        min_depth_range: float,
        default_on_surface_th: float,
    ) -> tuple[float, bool]:
        """Return a depth threshold if we have a bimodal depth distribution.

        If the depth values are in a large enough range (> min_depth_range) we may
        be looking at more than one surface within our patch. This could either be
        two disjoint surfaces of the object or the object and the background.

        To figure out if we have two disjoint sets of depth values we look at the
        histogram and check for empty bins in the middle. The center of the empty
        part if the histogram will be defined as the threshold.

        If we do have a bimodal depth distribution, we effectively have two surfaces.
        This could be the mug's handle vs the mug's body, or the front lip of a mug
        vs the rear of the mug. We will use the depth threshold defined above to mask
        out the surface that is not in the center of the patch.

        Args:
            depth_patch: sensor patch observations of depth
            semantic_patch: binary mask indicating on-object locations
            min_depth_range: minimum range of depth values to even be considered
            default_on_surface_th: default threshold to use if no bimodal distribution
                is found

        Returns:
            threshold and whether we want to use values above or below threshold
        """
        center_loc = (depth_patch.shape[0] // 2, depth_patch.shape[1] // 2)
        depth_center = depth_patch[center_loc[0], center_loc[1]]
        semantic_center = semantic_patch[center_loc[0], center_loc[1]]

        depths = np.asarray(depth_patch).flatten()
        flip_sign = False
        th = default_on_surface_th
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
                if depth_center > th and semantic_center > 0:
                    # if the FOV's center is on the further away surface and the FOV's
                    # center is on-object, then we want to use the further-away surface.
                    flip_sign = True

        return th, flip_sign

    def _get_surface_from_depth(
        self,
        depth_patch: npt.NDArray[np.float64],
        semantic_patch: npt.NDArray[np.int_],
        default_on_surface_th: float,
    ) -> npt.NDArray[np.int_]:
        """Return surface patch information from heuristics on depth patch.

        This function returns a binary mask indicating whether each pixel is
        "on-surface" using heuristics on the depth patch and the semantic mask.
        When we say "on-surface", we mean "on the part of the object that the sensor
        is centered on". For example, the sensor may be looking directly at the front
        lip of a mug, but the back lip of the mug is also in view. While both parts
        of the mug are on-object, we only want to use the front lip of the mug for
        later computation (such as surface normal extraction and curvature estimation),
        and so we want to mask out the back lip of the mug.

        Continuing with the mug front/back lip example, we separate the front
        and back lips by their depth data. When we generate a histogram of pixel
        depths, we should see two distinct peaks of the histogram (or 3 peaks if there
        is a part of the field of view that is off-object entirely). We would then
        compute which depth threshold separates the two peaks that correspond to the
        two surfaces, and this is performed by `get_on_surface_th`. This function
        uses that value to mask out the pixels that are beyond that threshold. Finally,
        we element-wise multiply this surface mask by the semantic (i.e., object mask)
        mask which ensures that off-object observations are also masked out.

        Note that we most often don't have multiple surfaces in view. For example,
        when exploring a mug, we are most often looking directly at one locally
        connected part of the mug, such as a patch on the mug's cylindrical body.
        In this case, we don't attempt to find a surface-separating threshold, and we
        instead use the default threshold `default_on_surface_th`. As with a
        computed threshold, anything beyond it is zeroed out. For surface agents,
        this threshold is usually quite small (e.g., 5 cm). For distant agents, the
        threshold is large (e.g., 1000 meters) which is functionally infinite.

        Args:
            depth_patch: sensor patch observations of depth
            semantic_patch: binary mask indicating on-object locations
            default_on_surface_th: default threshold to use if no bimodal distribution
                is found

        Returns:
            Sensor-patch-shaped info about whether each pixel is on surface or not.
        """
        # avoid large range when seeing the table (goes up to almost 100 and then
        # just using 8 bins will not work anymore)
        depth_patch = np.array(depth_patch)
        depth_patch[depth_patch > 1] = 1.0

        # If all depth values are at maximum (1.0), then we are automatically
        # off-object.
        if np.all(depth_patch >= 1.0):
            return np.zeros_like(depth_patch, dtype=bool)

        # Compute the on-surface depth threshold (and whether we need to flip the
        # sign), and apply it to the depth to get the semantic patch.
        th, flip_sign = self._get_on_surface_th(
            depth_patch,
            semantic_patch,
            min_depth_range=0.01,
            default_on_surface_th=default_on_surface_th,
        )
        if flip_sign is False:
            surface_patch = depth_patch < th
        else:
            surface_patch = depth_patch > th

        return surface_patch * semantic_patch
