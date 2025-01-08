import os

from tbp.monty.frameworks.run import main  # noqa: E402
from tbp.monty.frameworks.run_env import setup_env

# os.environ["saveplots"] = "True"

setup_env()
from configs import CONFIGS  # noqa: E402

main(all_configs=CONFIGS, experiments=["randrot_noise_sim_on_scan_monty_world"])

# from numbers import Number
# from typing import Tuple

# import matplotlib.pyplot as plt
# import numpy as np

# from tbp.monty.frameworks.environment_utils.transforms import DepthTo3DLocations


# def get_on_surface_th(
#     depth_patch,
#     min_depth_range: Number,
#     default_on_surface_th: Number,
# ) -> Tuple[Number, bool]:
#     """Return a depth threshold if we have a bimodal depth distribution.

#     If the depth values are in a large enough range (> min_depth_range) we may
#     be looking at more than one surface within our patch. This could either be
#     two disjoint surfaces of the object or the object and the background.

#     To figure out if we have two disjoint sets of depth values we look at the
#     histogram and check for empty bins in the middle. The center of the empty
#     part if the histogram will be defined as the threshold.

#     Next, we want to check if we should use the depth values above or below the
#     threshold. Currently this is done by looking which side of the distribution
#     is larger (occupies more space in the patch). Alternatively we could check
#     which side the depth at the center of the patch is on. I'm not sure what would
#     be better.

#     Lastly, if we do decide to use the depth points that are further away, we need
#     to make sure they are not the points that are off the object. Currently this is
#     just done with a simple heuristic (depth difference < 0.1) but in the future we
#     will probably have to find a better solution for this.

#     Args:
#         depth_patch: sensor patch observations of depth
#         min_depth_range: minimum range of depth values to even be considered
#         default_on_surface_th: default threshold to use if no bimodal distribution
#             is found
#     Returns:
#         threshold and whether we want to use values above or below threshold
#     """
#     depths = np.asarray(depth_patch).flatten()
#     flip_sign = False
#     th = default_on_surface_th
#     if (max(depths) - min(depths)) > min_depth_range:
#         # only check for bimodal distribution if we have a large enough
#         # range in depth values
#         height, bins = np.histogram(
#             np.array(depth_patch).flatten(), bins=8, density=False
#         )
#         gap = np.where(height == 0)[0]
#         if len(gap) > 0:
#             # There is a bimodal distribution
#             gap_center = len(gap) // 2
#             th_id = gap[gap_center]
#             th = bins[th_id]
#             # Check which side of the distribution we should use
#             if np.sum(height[:th_id]) < np.sum(height[th_id:]):
#                 # more points in the patch are on the further away surface
#                 if (bins[-1] - bins[0]) < 0.1:
#                     # not too large distance between depth values -> avoid
#                     # flipping sign when off object
#                     flip_sign = True
#     return th, flip_sign


# def get_semantic_from_depth(
#     depth_patch: np.ndarray,
#     default_on_surface_th: Number,
# ) -> np.ndarray:
#     """Return semantic patch information from heuristics on depth patch.

#     Args:
#         depth_patch: sensor patch observations of depth
#         default_on_surface_th: default threshold to use if no bimodal distribution
#             is found
#     Returns:
#         sensor patch shaped info about whether each pixel is on surface of not
#     """
#     # avoid large range when seeing the table (goes up to almost 100 and then
#     # just using 8 bins will not work anymore)
#     depth_patch = np.array(depth_patch)
#     depth_patch[depth_patch > 1] = 1.0

#     # If all depth values are at maximum (1.0), then we are automatically
#     # off-object.
#     if np.all(depth_patch == 1.0):
#         return np.zeros_like(depth_patch, dtype=bool)

#     # Compute the on-suface depth threshold (and whether we need to flip the
#     # sign), and apply it to the depth to get the semantic patch.
#     th, flip_sign = get_on_surface_th(
#         depth_patch,
#         min_depth_range=0.01,
#         default_on_surface_th=default_on_surface_th,
#     )
#     print("flip_sign", flip_sign)
#     if flip_sign is False:
#         semantic_patch = depth_patch < th
#     else:
#         semantic_patch = depth_patch > th
#     return semantic_patch


# depth_path = "/Users/sknudstrup/depth.npy"
# depth = np.load(depth_path)
# # t = DepthTo3DLocations()
# clip_value = 0.05
# default_on_surface_th = clip_value

# # semantic = get_semantic_from_depth(depth, default_on_surface_th)
# # plt.imshow(semantic)
# # plt.colorbar()
# # plt.show()
# depth_patch = depth.copy()
# min_depth_range = 0.01
# depths = np.asarray(depth_patch).flatten()
# flip_sign = False
# th = default_on_surface_th
# if (max(depths) - min(depths)) > min_depth_range:
#     # only check for bimodal distribution if we have a large enough
#     # range in depth values
#     height, bins = np.histogram(np.array(depth_patch).flatten(), bins=8, density=False)
#     gap = np.where(height == 0)[0]
#     if len(gap) > 0:
#         # There is a bimodal distribution
#         gap_center = len(gap) // 2
#         th_id = gap[gap_center]
#         th = bins[th_id]
#         # Check which side of the distribution we should use
#         if np.sum(height[:th_id]) < np.sum(height[th_id:]):
#             # more points in the patch are on the further away surface
#             if (bins[-1] - bins[0]) < 0.1:
#                 # not too large distance between depth values -> avoid
#                 # flipping sign when off object
#                 flip_sign = True
