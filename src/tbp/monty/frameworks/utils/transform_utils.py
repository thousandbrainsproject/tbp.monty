# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import numpy as np
import quaternion  # noqa: F401; required by np.quaternion


def find_transform_instance(composed_transform, transform_type):
    """Find the first instance of transform_type.

    Loop over a composition of transforms and return the first instance of
    transform_type.

    Args:
        composed_transform: Output of T.Compose([args]), T being torchvision or
            torch_geometric transforms
        transform_type: The class you are looking for an instance of

    Returns:
        The first instance of transform_type or None if not found
    """
    # if compsoed transform is not a composition but already transform type, just return
    if isinstance(composed_transform, transform_type):
        return composed_transform

    instance = None
    for transform in composed_transform.transforms:
        if isinstance(transform, transform_type):
            instance = transform

    if not instance:
        print(f"Warning, no {transform_type} instance found")

    return instance


def numpy_to_scipy_quat(quat):
    """Convert from wxyz to xyzw format of quaternions.

    i.e. identity rotation in scipy is (0,0,0,1).

    Args:
        quat: A quaternion in wxyz format

    Returns:
        A quaternion in xyzw format
    """
    new_quat = np.array((quat[1], quat[2], quat[3], quat[0]))

    return new_quat


def scipy_to_numpy_quat(quat: np.ndarray) -> np.quaternion:
    numpy_quat = np.quaternion(quat[3], quat[0], quat[1], quat[2])
    return numpy_quat
