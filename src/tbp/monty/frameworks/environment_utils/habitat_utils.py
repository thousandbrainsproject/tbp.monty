# Copyright 2025 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import numpy as np


def get_bounding_corners(object_ref):
    """Determine and return the bounding box of a Habitat object.

    Determines and returns the bounding box (defined by a "max" and "min" corner) of
    a Habitat object (such as a mug), given in world coordinates.

    Specifically uses the "axis-aligned bounding box" (aabb) available in Habitat; this
    is a bounding box aligned with the axes of the co-oridante system, which tends to
    be computationally efficient to retrieve.

    Args:
        object_ref : the Habitat object instance

    Returns:
        Two np.arrays : min_corner and max_corner, the defining corners of the bounding
        box
    """
    object_aabb = object_ref.collision_shape_aabb

    # The bounding box will be in the coordinate frame of the object, and so needs to be
    # transformed (rotated and translated) based on the pose of the object in the
    # environment
    # The matrix returned by object_ref.transformation can apply this transformation
    # pointwise to the min and max corner points below
    object_t_mat = object_ref.transformation

    min_corner = object_aabb.min
    max_corner = object_aabb.max

    min_corner = np.array(object_t_mat.transform_point(min_corner))
    max_corner = np.array(object_t_mat.transform_point(max_corner))

    return min_corner, max_corner
