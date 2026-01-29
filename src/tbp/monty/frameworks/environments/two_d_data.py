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
import time
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import PIL
import quaternion as qt
from scipy.ndimage import gaussian_filter

from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environment_utils.transforms import DepthTo3DLocations
from tbp.monty.frameworks.environments.environment import SimulatedEnvironment
from tbp.monty.frameworks.models.abstract_monty_classes import (
    AgentObservations,
    Observations,
    SensorObservations,
)
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    ProprioceptiveState,
    SensorState,
)
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.path import monty_data_path

logger = logging.getLogger(__name__)

__all__ = [
    "OmniglotEnvironment",
    "SaccadeOnImageEnvironment",
    "SaccadeOnImageFromStreamEnvironment",
]


class OmniglotEnvironment(SimulatedEnvironment):
    """Environment for Omniglot dataset."""

    def __init__(self, patch_size=10, data_path=None):
        """Initialize environment.

        Args:
            patch_size: height and width of patch in pixels, defaults to 10
            data_path: path to the omniglot dataset. If None, defaults to
                ~/tbp/data/omniglot/python/
        """
        self.patch_size = patch_size
        # Letters are always presented upright
        self.rotation = qt.from_rotation_vector([np.pi / 2, 0.0, 0.0])
        self.step_num = 0
        self.state = 0
        self.data_path = monty_data_path(data_path, "omniglot/python")
        alphabet_path = self.data_path / "images_background"
        self.alphabet_names = [a.name for a in sorted(alphabet_path.glob("[!.]*"))]
        self.current_alphabet = self.alphabet_names[0]
        self.character_id = 1
        self.character_version = 1

        self.current_image, self.locations = self.load_new_character_data()
        # Just for compatibility. TODO: find cleaner way to do this.
        self._agents = [type("FakeAgent", (object,), {"action_space_type": "2d"})()]

    def step(
        self, actions: Sequence[Action]
    ) -> tuple[Observations, ProprioceptiveState]:
        """Retrieve the next observation.

        Since the omniglot dataset includes stroke information (the order in which
        the character was drawn as a list of x,y coordinates) we use that for movement.
        This means we start at the first x,y coordinate saved in the move path and then
        move in increments specified by amount through this list. Overall there are
        usually several hundred points (~200-400) but it varies between characters and
        versions.
        If we reach the end of a move path and the episode is not finished, we start
        from the beginning again. If len(move_path) % amount != 0, we will sample
        different points on the second pass.

        Args:
            actions: Not used at the moment since we just follow the draw path. However,
            we do use the rotation_degrees to determine the amount of pixels to move at
            each step.

        Returns:
            The observations and proprioceptive state.
        """
        obs = self._observation()

        for action in actions:
            amount = 1
            if hasattr(action, "rotation_degrees"):
                amount = max(action.rotation_degrees, 1)
            self.step_num += int(amount)
            obs = self._observation()

        return obs

    def _observation(self) -> Observations:
        query_loc = self.locations[self.step_num % self.max_steps]
        patch = self.get_image_patch(
            self.current_image,
            query_loc,
            self.patch_size,
        )
        depth = 1.2 - gaussian_filter(np.array(~patch, dtype=float), sigma=0.5)
        obs = Observations(
            {
                AgentID("agent_id_0"): AgentObservations(
                    {
                        SensorID("patch"): SensorObservations(
                            {
                                "depth": depth,
                                "semantic": np.array(~patch, dtype=int),
                                "rgba": np.stack([depth, depth, depth], axis=2),
                            }
                        ),
                        SensorID("view_finder"): SensorObservations(
                            {
                                "depth": self.current_image,
                                "semantic": np.array(~patch, dtype=int),
                            }
                        ),
                    }
                )
            }
        )
        return obs, self.get_state()

    def get_state(self) -> ProprioceptiveState:
        loc = self.locations[self.step_num % self.max_steps]
        sensor_position = np.array([loc[0], loc[1], 0])
        return ProprioceptiveState(
            {
                AgentID("agent_id_0"): AgentState(
                    sensors={
                        SensorID("patch" + ".depth"): SensorState(
                            rotation=self.rotation,
                            position=sensor_position,
                        ),
                        SensorID("patch" + ".rgba"): SensorState(
                            rotation=self.rotation,
                            position=sensor_position,
                        ),
                        SensorID("view_finder" + ".depth"): SensorState(
                            rotation=self.rotation,
                            position=sensor_position,
                        ),
                        SensorID("view_finder" + ".rgba"): SensorState(
                            rotation=self.rotation,
                            position=sensor_position,
                        ),
                    },
                    rotation=self.rotation,
                    position=np.array([0, 0, 0]),
                )
            }
        )

    def switch_to_object(self, alphabet_id, character_id, version_id):
        self.current_alphabet = self.alphabet_names[alphabet_id]
        self.character_id = character_id
        self.character_version = version_id
        self.current_image, self.locations = self.load_new_character_data()

    def reset(self) -> tuple[Observations, ProprioceptiveState]:
        self.step_num = 0
        patch = self.get_image_patch(
            self.current_image, self.locations[self.step_num], self.patch_size
        )
        depth = 1.2 - gaussian_filter(np.array(~patch, dtype=float), sigma=0.5)
        obs = Observations(
            {
                AgentID("agent_id_0"): AgentObservations(
                    {
                        SensorID("patch"): SensorObservations(
                            {
                                "depth": depth,
                                "semantic": np.array(~patch, dtype=int),
                                "rgba": np.stack([depth, depth, depth], axis=2),
                            }
                        ),
                        SensorID("view_finder"): SensorObservations(
                            {
                                "depth": self.current_image,
                                "semantic": np.array(~patch, dtype=int),
                            }
                        ),
                    }
                )
            }
        )
        return obs, self.get_state()

    def load_new_character_data(self):
        img_char_dir = (
            self.data_path
            / "images_background"
            / self.current_alphabet
            / f"character{self.character_id:02d}"
        )
        stroke_char_dir = (
            self.data_path
            / "strokes_background"
            / self.current_alphabet
            / f"character{self.character_id:02d}"
        )
        first_img_char_child = next(img_char_dir.iterdir()).name
        char_img_names = first_img_char_child.split("_")[0]
        char_stem = f"{char_img_names}_{self.character_version:02d}"
        img_file = img_char_dir / f"{char_stem}.png"
        current_image = load_img(img_file)
        move_path = load_motor(stroke_char_dir / f"{char_stem}.txt")
        logger.info(f"Finished loading new image from {img_file}")
        locations = self.motor_to_locations(move_path)
        maxloc = current_image.shape[0] - self.patch_size
        # Don't use locations at the border where patch doesn't fit anymore
        locs_in_range = np.where(
            (locations[:, 0] > self.patch_size)
            & (locations[:, 1] > self.patch_size)
            & (locations[:, 0] < maxloc)
            & (locations[:, 1] < maxloc)
        )
        locations = locations[locs_in_range]
        self.max_steps = len(locations) - 1
        return current_image, locations

    def get_image_patch(self, img, loc, patch_size):
        loc = np.array(loc, dtype=int)
        startx = loc[1] - patch_size // 2
        stopx = loc[1] + patch_size // 2
        starty = loc[0] - patch_size // 2
        stopy = loc[0] + patch_size // 2
        return img[startx:stopx, starty:stopy]

    def motor_to_locations(self, motor):
        motor = [d[:, 0:2] for d in motor]
        motor = [space_motor_to_img(d) for d in motor]
        locations = np.zeros(2)
        for stroke in motor:
            locations = np.vstack([locations, stroke])
        return locations[1:]

    def close(self) -> None:
        self._current_state = None


class SaccadeOnImageEnvironment(SimulatedEnvironment):
    """Environment for moving over a 2D image with depth channel.

    Images should be stored in .png format for rgb and .data format for depth.
    """

    def __init__(self, patch_size=64, data_path=None):
        """Initialize environment.

        Args:
            patch_size: height and width of patch in pixels, defaults to 64
            data_path: path to the image dataset. If None, defaults to
                ~/tbp/data/worldimages/labeled_scenes/
        """
        self.patch_size = patch_size
        # Images are always presented upright so patch and agent rotation is always
        # the same. Since we don't use this, value doesn't matter much.
        self.rotation = qt.from_rotation_vector([np.pi / 2, 0.0, 0.0])
        self.state = 0
        self.data_path = monty_data_path(data_path, "worldimages/labeled_scenes")
        self.scene_names = [a.name for a in sorted(self.data_path.glob("[!.]*"))]
        self.current_scene = self.scene_names[0]
        self.scene_version = 0

        (
            self.current_depth_image,
            self.current_rgb_image,
            self.current_loc,
        ) = self.load_new_scene_data()
        self.move_area = self.get_move_area()

        # Get 3D scene point cloud array from depth image
        (
            self.current_scene_point_cloud,
            self.current_sf_scene_point_cloud,
        ) = self.get_3d_scene_point_cloud()

        # Just for compatibility. TODO: find cleaner way to do this.
        self._agents = [
            type(
                "FakeAgent",
                (object,),
                {"action_space_type": "distant_agent_no_translation"},
            )()
        ]

        # Instantiate once and reuse when checking action name in step()
        # TODO Use 2D-specific actions instead of overloading? Habitat actions
        self._valid_actions = ["look_up", "look_down", "turn_left", "turn_right"]

    def step(
        self, actions: Sequence[Action]
    ) -> tuple[Observations, ProprioceptiveState]:
        """Retrieve the next observation.

        Args:
            actions: moving up, down, left or right from current location.

        Returns:
            The observation and proprioceptive state.
        """
        obs = self._observation()

        for action in actions:
            if action.name in self._valid_actions:
                amount = action.rotation_degrees
            else:
                amount = 0

            if np.abs(amount) < 1:
                amount = 1
            # Make sure amount is int since we are moving using pixel indices
            amount = int(amount)
            self.current_loc = self.get_next_loc(action.name, amount)
            obs = self._observation()

        return obs

    def _observation(self) -> Observations:
        (
            depth_patch,
            rgb_patch,
            depth3d_patch,
            sensor_frame_patch,
        ) = self.get_image_patch(self.current_loc)
        obs = Observations(
            {
                AgentID("agent_id_0"): AgentObservations(
                    {
                        SensorID("patch"): SensorObservations(
                            {
                                "depth": depth_patch,
                                "rgba": rgb_patch,
                                "semantic_3d": depth3d_patch,
                                "sensor_frame_data": sensor_frame_patch,
                                "world_camera": self.world_camera,
                                # Save pixel loc for plotting
                                "pixel_loc": self.current_loc,
                            }
                        ),
                        SensorID("view_finder"): SensorObservations(
                            {
                                "depth": self.current_depth_image,
                                "rgba": self.current_rgb_image,
                            }
                        ),
                    }
                )
            }
        )
        return obs, self.get_state()

    def get_state(self) -> ProprioceptiveState:
        loc = self.current_loc
        # Provide LM w/ sensor position in 3D, body-centric coordinates
        # instead of pixel indices
        sensor_position = self.get_3d_coordinates_from_pixel_indices(loc[:2])

        # NOTE: This is super hacky and only works for 1 agent with 1 sensor
        return ProprioceptiveState(
            {
                AgentID("agent_id_0"): AgentState(
                    sensors={
                        SensorID("patch" + ".depth"): SensorState(
                            rotation=self.rotation, position=sensor_position
                        ),
                        SensorID("patch" + ".rgba"): SensorState(
                            rotation=self.rotation, position=sensor_position
                        ),
                        SensorID("view_finder" + ".depth"): SensorState(
                            rotation=self.rotation,
                            position=sensor_position,
                        ),
                        SensorID("view_finder" + ".rgba"): SensorState(
                            rotation=self.rotation,
                            position=sensor_position,
                        ),
                    },
                    rotation=self.rotation,
                    position=np.array([0, 0, 0]),
                )
            }
        )

    def switch_to_object(self, scene_id, scene_version_id):
        """Load new image to be used as environment."""
        self.current_scene = self.scene_names[scene_id]
        self.scene_version = scene_version_id
        (
            self.current_depth_image,
            self.current_rgb_image,
            self.current_loc,
        ) = self.load_new_scene_data()

        # Get 3D scene point cloud array from depth image
        (
            self.current_scene_point_cloud,
            self.current_sf_scene_point_cloud,
        ) = self.get_3d_scene_point_cloud()

    def reset(self) -> tuple[Observations, ProprioceptiveState]:
        """Reset environment and extract image patch.

        TODO: clean up. Do we need this? No reset is required in this environment
          interface, so this should be indicated more clearly.

        Returns:
            The observation from the image patch.
        """
        (
            depth_patch,
            rgb_patch,
            depth3d_patch,
            sensor_frame_patch,
        ) = self.get_image_patch(
            self.current_loc,
        )
        obs = Observations(
            {
                AgentID("agent_id_0"): AgentObservations(
                    {
                        SensorID("patch"): SensorObservations(
                            {
                                "depth": depth_patch,
                                "rgba": rgb_patch,
                                "semantic_3d": depth3d_patch,
                                "sensor_frame_data": sensor_frame_patch,
                                "world_camera": self.world_camera,
                                "pixel_loc": np.array(self.current_loc),
                            }
                        ),
                        SensorID("view_finder"): SensorObservations(
                            {
                                "depth": self.current_depth_image,
                                "rgba": self.current_rgb_image,
                            }
                        ),
                    }
                )
            }
        )
        return obs, self.get_state()

    def load_new_scene_data(self):
        """Load depth and rgb data for next scene environment.

        Returns:
            current_depth_image: The depth image.
            current_rgb_image: The RGB image.
            start_location: The start location.
        """
        # Set data paths
        current_depth_path = (
            self.data_path
            / f"{self.current_scene}"
            / f"depth_{self.scene_version}.data"
        )
        current_rgb_path = (
            self.data_path / f"{self.current_scene}" / f"rgb_{self.scene_version}.png"
        )
        # Load & process data
        current_rgb_image = self.load_rgb_data(current_rgb_path)
        height, width, _ = current_rgb_image.shape
        current_depth_image = self.load_depth_data(current_depth_path, height, width)
        current_depth_image = self.process_depth_data(current_depth_image)
        # set start location to center of image
        # TODO: find object if not in center
        obs_shape = current_depth_image.shape
        start_location = [obs_shape[0] // 2, obs_shape[1] // 2]
        return current_depth_image, current_rgb_image, start_location

    def load_depth_data(self, depth_path, height, width):
        """Load depth image from .data file.

        Returns:
            The depth image.
        """
        return np.fromfile(depth_path, np.float32).reshape(height, width)

    def process_depth_data(self, depth):
        """Process depth data by reshaping, clipping and flipping.

        Returns:
            The processed depth image.
        """
        # Set nan values to 10m
        depth[np.isnan(depth)] = 10

        depth_clipped = depth.copy()
        # Anything that's further away than 40cm is clipped
        # TODO: make this a hyperparameter?
        depth_clipped[depth > 0.4] = 10
        # flipping image makes visualization more intuitive. If we want to have this
        # in here we also have to comment in the flipping in the rgb image and probably
        # flip left-right. It may be better to flip the image in the app, depending on
        # sensor orientation (TODO).
        return depth_clipped  # np.flipud(depth_clipped)

    def load_rgb_data(self, rgb_path):
        """Load RGB image and put into np array.

        Returns:
            The RGB image.
        """
        return np.array(
            PIL.Image.open(rgb_path)  # .transpose(PIL.Image.FLIP_TOP_BOTTOM)
        )

    def get_3d_scene_point_cloud(self):
        """Turn 2D depth image into 3D pointcloud using DepthTo3DLocations.

        This point cloud is used to estimate the sensor displacement in 3D space
        between two subsequent steps. Without this we get displacements in pixel
        space which does not work with our 3D models.

        Returns:
            current_scene_point_cloud: The 3D scene point cloud.
            current_sf_scene_point_cloud: The 3D scene point cloud in sensor frame.
        """
        agent_id = AgentID("agent_01")
        sensor_id = SensorID("patch_01")
        obs = Observations(
            {
                agent_id: AgentObservations(
                    {sensor_id: SensorObservations({"depth": self.current_depth_image})}
                )
            }
        )
        rotation = qt.from_rotation_vector([np.pi / 2, 0.0, 0.0])
        state = ProprioceptiveState(
            {
                agent_id: AgentState(
                    sensors={
                        SensorID(sensor_id + ".depth"): SensorState(
                            rotation=rotation,
                            position=np.array([0, 0, 0]),
                        )
                    },
                    rotation=rotation,
                    position=np.array([0, 0, 0]),
                )
            }
        )

        # Apply gaussian smoothing transform to depth image
        # Uncomment line below and add import, if needed
        # transform = GaussianSmoothing(agent_id=agent_id, sigma=2, kernel_width=3)
        # obs = transform.call(obs)

        transform = DepthTo3DLocations(
            agent_id=agent_id,
            sensor_ids=[sensor_id],
            resolutions=[self.current_depth_image.shape],
            world_coord=True,
            zooms=1,
            # hfov of iPad front camera from
            # https://developer.apple.com/library/archive/documentation/DeviceInformation/Reference/iOSDeviceCompatibility/Cameras/Cameras.html
            # TODO: determine dynamically which device is sending data
            hfov=54.201,
            get_all_points=True,
            use_semantic_sensor=False,
            depth_clip_sensors=[0],
            clip_value=1.1,
        )
        obs_3d = transform.call(obs, state=state)
        current_scene_point_cloud = obs_3d[agent_id][sensor_id]["semantic_3d"]
        image_shape = self.current_depth_image.shape
        current_scene_point_cloud = current_scene_point_cloud.reshape(
            (image_shape[0], image_shape[1], 4)
        )
        current_sf_scene_point_cloud = obs_3d[agent_id][sensor_id]["sensor_frame_data"]
        current_sf_scene_point_cloud = current_sf_scene_point_cloud.reshape(
            (image_shape[0], image_shape[1], 4)
        )
        self.world_camera = obs_3d[agent_id][sensor_id]["world_camera"]
        return current_scene_point_cloud, current_sf_scene_point_cloud

    def get_3d_coordinates_from_pixel_indices(self, pixel_idx):
        """Retrieve 3D coordinates of a pixel.

        Returns:
            The 3D coordinates of the pixel.
        """
        [i, j] = pixel_idx
        return np.array(self.current_scene_point_cloud[i, j, :3])

    def get_move_area(self):
        """Calculate area in which patch can move on the image.

        Returns:
            The move area.
        """
        obs_shape = self.current_depth_image.shape
        half_patch_size = self.patch_size // 2 + 1
        return np.array(
            [
                [half_patch_size, obs_shape[0] - half_patch_size],
                [half_patch_size, obs_shape[1] - half_patch_size],
            ]
        )

    def get_next_loc(self, action_name, amount):
        """Calculate next location in pixel space given the current action.

        Returns:
            The next location in pixel space.
        """
        new_loc = np.array(self.current_loc)
        if action_name == "look_up":
            new_loc[0] -= amount
        elif action_name == "look_down":
            new_loc[0] += amount
        elif action_name == "turn_left":
            new_loc[1] -= amount
        elif action_name == "turn_right":
            new_loc[1] += amount
        else:
            logger.error(f"{action_name} is not a valid action, not moving.")
        # Make sure location stays within move area
        if new_loc[0] < self.move_area[0][0]:
            new_loc[0] = self.move_area[0][0]
        elif new_loc[0] > self.move_area[0][1]:
            new_loc[0] = self.move_area[0][1]
        if new_loc[1] < self.move_area[1][0]:
            new_loc[1] = self.move_area[1][0]
        elif new_loc[1] > self.move_area[1][1]:
            new_loc[1] = self.move_area[1][1]
        return new_loc

    def get_image_patch(self, loc):
        """Extract 2D image patch from a location in pixel space.

        Returns:
            depth_patch: The depth patch.
            rgb_patch: The rgb patch.
            depth3d_patch: The depth3d patch.
            sensor_frame_patch: The sensor frame patch.
        """
        loc = np.array(loc, dtype=int)
        x_start = loc[0] - self.patch_size // 2
        x_stop = loc[0] + self.patch_size // 2
        y_start = loc[1] - self.patch_size // 2
        y_stop = loc[1] + self.patch_size // 2
        depth_patch = self.current_depth_image[x_start:x_stop, y_start:y_stop]
        rgb_patch = self.current_rgb_image[x_start:x_stop, y_start:y_stop]
        depth3d_patch = self.current_scene_point_cloud[x_start:x_stop, y_start:y_stop]
        depth_shape = depth3d_patch.shape
        depth3d_patch = depth3d_patch.reshape(
            (depth_shape[0] * depth_shape[1], depth_shape[2])
        )
        sensor_frame_patch = self.current_sf_scene_point_cloud[
            x_start:x_stop, y_start:y_stop
        ]
        sensor_frame_patch = sensor_frame_patch.reshape(
            (depth_shape[0] * depth_shape[1], depth_shape[2])
        )

        assert (
            depth_patch.shape[0] * depth_patch.shape[1]
            == self.patch_size * self.patch_size
        ), f"Didn't extract a patch of size {self.patch_size}"
        return depth_patch, rgb_patch, depth3d_patch, sensor_frame_patch

    def close(self) -> None:
        self._current_state = None


class SaccadeOnImageFromStreamEnvironment(SaccadeOnImageEnvironment):
    """Environment for moving over a 2D streamed image with depth channel."""

    def __init__(self, patch_size=64, data_path=None):
        """Initialize environment.

        Args:
            patch_size: height and width of patch in pixels, defaults to 64
            data_path: path to the image dataset. If None, defaults to
                ~/tbp/data/worldimages/world_data_stream/
        """
        # TODO: use super() to avoid repeating lines of code
        self.patch_size = patch_size
        # Letters are always presented upright
        self.rotation = qt.from_rotation_vector([np.pi / 2, 0.0, 0.0])
        self.state = 0
        self.data_path = monty_data_path(data_path, "worldimages/world_data_stream")
        self.scene_names = [a.name for a in sorted(self.data_path.glob("[!.]*"))]
        self.current_scene = 0

        (
            self.current_depth_image,
            self.current_rgb_image,
            self.current_loc,
        ) = self.load_new_scene_data()
        self.move_area = self.get_move_area()

        # Get 3D scene point cloud array from depth image (in world rf and sensor rf)
        (
            self.current_scene_point_cloud,
            self.current_sf_scene_point_cloud,
        ) = self.get_3d_scene_point_cloud()

        # Just for compatibility. TODO: find cleaner way to do this.
        self._agents = [
            type(
                "FakeAgent",
                (object,),
                {"action_space_type": "distant_agent_no_translation"},
            )()
        ]

        # Instantiate once and reuse when checking action name in step()
        # TODO Use 2D-specific actions instead of overloading? Habitat actions
        # TODO Fix how inheritance is used here. We duplicate the below code because we
        #      don't call super().__init__ while inheriting
        self._valid_actions = ["look_up", "look_down", "turn_left", "turn_right"]

    def switch_to_scene(self, scene_id):
        self.current_scene = scene_id
        (
            self.current_depth_image,
            self.current_rgb_image,
            self.current_loc,
        ) = self.load_new_scene_data()

        # Get 3D scene point cloud array from depth image
        (
            self.current_scene_point_cloud,
            self.current_sf_scene_point_cloud,
        ) = self.get_3d_scene_point_cloud()

    def load_new_scene_data(self):
        current_depth_path = self.data_path / f"depth_{self.current_scene}.data"
        current_rgb_path = self.data_path / f"rgb_{self.current_scene}.png"
        # Load rgb image
        wait_count = 0
        while not current_rgb_path.exists():
            if wait_count % 10 == 0:
                # Print every 10 seconds
                print("Waiting for new RGBD data...")
            time.sleep(1)
            wait_count += 1

        load_succeeded = False
        while not load_succeeded:
            try:
                current_rgb_image = self.load_rgb_data(current_rgb_path)
                load_succeeded = True
            except PIL.UnidentifiedImageError:
                print("waiting for rgb file to finish streaming")
                time.sleep(1)
        height, width, _ = current_rgb_image.shape

        # Load depth image
        while not current_depth_path.exists():
            print(f"Waiting for new depth data. Looking for {current_depth_path}")
            time.sleep(1)
        load_succeeded = False
        while not load_succeeded:
            try:
                current_depth_image = self.load_depth_data(
                    current_depth_path, height, width
                )
                load_succeeded = True
            except ValueError:
                print("waiting for depth file to finish streaming")
                time.sleep(1)
        current_depth_image = self.process_depth_data(current_depth_image)

        # set start location to center of image
        # TODO: find object if not in center
        start_location = [height // 2, width // 2]
        return current_depth_image, current_rgb_image, start_location


# Functions from omniglot/python.demo.py
# TODO: integrate better and maybe rewrite
def load_img(fn):
    img = plt.imread(fn)
    return np.array(img, dtype=bool)


def load_motor(fn):
    motor = []
    with fn.open() as fid:
        lines = fid.readlines()
    lines = [line.strip() for line in lines]
    for myline in lines:
        if myline == "START":  # beginning of character
            stk = []
        elif myline == "BREAK":  # break between strokes
            stk = np.array(stk)
            motor.append(stk)  # add to list of strokes
            stk = []
        else:
            arr = np.fromstring(myline, dtype=float, sep=",")
            stk.append(arr)
    return motor


def space_motor_to_img(pt):
    pt[:, 1] = -pt[:, 1]
    return pt
