# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""A collection of plot utilities used during normal platform runtime."""

from __future__ import annotations

import numpy as np
from skimage.transform import resize


def mark_obs(vis_obs, patch_obs):
    """Mark vis_obs with the observations from a patch.

    Returns:
        Marked observations.
    """
    marked_obs = np.array(vis_obs["rgba"].copy())
    marked_obs[29:35, 29:35] = [0, 0, 255, 255]
    patch_obs_rgba = np.array(patch_obs["rgba"])[:, :, :3] / 255
    resized_patch_obs = resize(patch_obs_rgba, (10, 10, 3))
    resized_patch_obs = np.array(resized_patch_obs * 255, dtype=int)
    rgba_patch_resized = np.insert(resized_patch_obs, 3, 255, axis=2)
    marked_obs[:10, :10] = rgba_patch_resized
    marked_obs[10, :11] = [0, 0, 255, 255]
    marked_obs[21, :11] = [0, 0, 255, 255]
    marked_obs[:21, 10] = [0, 0, 255, 255]
    depth_obs = np.array(patch_obs["depth"])
    depth_obs = resize(depth_obs, (10, 10))
    d_scaled = (depth_obs - depth_obs.min()) / (depth_obs.max() - depth_obs.min())
    patch_obs_depth = np.rollaxis(
        np.stack(
            [
                d_scaled * 255,
                d_scaled * 255,
                d_scaled * 255,
                np.ones(d_scaled.shape) * 255,
            ]
        ),
        0,
        3,
    )
    patch_obs_depth = np.array(patch_obs_depth, dtype=int)
    marked_obs[11:21, :10] = patch_obs_depth
    return marked_obs


def add_patch_outline_to_view_finder(view_finder_image, center_pixel_id, patch_size):
    # Calculate top-left and bottom-right coordinates of the square
    # Careful here: x coordinates and y coordinates in pixel space are inverted
    # i.e. x is index 1, y is index 0
    x1 = center_pixel_id[0] - patch_size // 2
    y1 = center_pixel_id[1] - patch_size // 2
    x2 = center_pixel_id[0] + patch_size // 2
    y2 = center_pixel_id[1] + patch_size // 2
    marked_image = np.array(view_finder_image.copy())

    # Set the outline pixels to blue
    marked_image[x1 : x1 + 2, y1:y2] = [0, 0, 255, 0]
    marked_image[x2 : x2 + 2, y1:y2] = [0, 0, 255, 0]
    marked_image[x1:x2, y1 : y1 + 2] = [0, 0, 255, 0]
    marked_image[x1:x2, y2 : y2 + 2] = [0, 0, 255, 0]
    return marked_image

    # TODO when seperating detection and logging from this plotting code, re-use
    # below
    # return (calculate_tpr(detection_results["true_positive"],
    #   detection_results["false_negative"]),
    #     calculate_fpr(detection_results["false_positive"],
    #   detection_results["true_negative"]))
