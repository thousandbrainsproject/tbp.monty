# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import json
from dataclasses import dataclass, fields
from json import JSONDecodeError
from pathlib import Path

from tbp.monty.math import IDENTITY_QUATERNION, ZERO_VECTOR, QuaternionWXYZ, VectorXYZ


class InvalidObjectMetadata(Exception):
    """The object metadata could not be decoded."""


@dataclass
class ObjectMetadata:
    """Contains the metadata for initializing a custom object."""

    refpos: VectorXYZ = ZERO_VECTOR
    """Reference position relative to which the 3D vertex coordinates are defined.
    This vector is subtracted from the positions."""

    refquat: QuaternionWXYZ = IDENTITY_QUATERNION
    """Reference orientation relative to which the 3D vertex coordinates and normals
    are defined. The conjugate of this quaternion is used to rotate the positions and
    normals. The model compiler normalizes the quaternion automatically.
    """


class ObjectMetadataDecoder(json.JSONDecoder):
    """Decodes custom object metadata from JSON.

    Expects a JSON object with the following structure:
    {
      "refpos": [0.0, 0.0, 0.0],
      "refquat": [1.0, 0.0, 0.0, 0.0],
    }
    """

    def __init__(self, object_type: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.object_type = object_type

    def decode(self, s) -> ObjectMetadata:
        try:
            decoded = super().decode(s)
        except JSONDecodeError as e:
            raise InvalidObjectMetadata(
                f"Couldn't decode '{self.object_type}' metadata."
            ) from e

        # Check for extra or missing fields in the metadata
        fields_set = set(decoded.keys())
        expected_fields = {f.name for f in fields(ObjectMetadata)}
        missing_fields = expected_fields - fields_set
        extra_fields = fields_set - expected_fields
        errors = []
        if missing_fields:
            errors.append(f"missing fields: {missing_fields}")
        if extra_fields:
            errors.append(f"extra fields: {extra_fields}")
        if errors:
            errors_str = " and ".join(errors)
            raise InvalidObjectMetadata(
                f"Object '{self.object_type}' metadata has {errors_str}"
            )

        return ObjectMetadata(**decoded)


def load_object_metadata(file_path: Path, object_type: str) -> ObjectMetadata:
    """Loads object metadata from a JSON file.

    Returns:
        ObjectMetadata
    """
    return json.load(
        file_path.open(),
        cls=ObjectMetadataDecoder,
        object_type=object_type,
    )
