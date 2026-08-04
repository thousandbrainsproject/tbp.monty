# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import json
from textwrap import dedent
from unittest import TestCase

import pytest

from tbp.monty.simulators.mujoco.objects import (
    InvalidObjectMetadata,
    ObjectMetadata,
    ObjectMetadataDecoder,
)


def load_object_metadata(json_str: str, object_type: str) -> ObjectMetadata:
    """Loads object metadata from a JSON file.

    Returns:
        ObjectMetadata
    """
    return json.loads(
        json_str,
        cls=ObjectMetadataDecoder,
        object_type=object_type,
    )


class ObjectMetadataDecoderTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.valid_json = dedent("""
        {
          "refpos": [1.0, 1.0, 1.0],
          "refquat": [1.0, 1.0, 0.0, 0.0]
        }""")
        cls.invalid_json = "{"
        cls.missing_field_json = dedent("""
        {
          "refpos": [1.0, 1.0, 1.0]
        }
        """)
        cls.extra_field_json = dedent("""
        {
          "refpos": [1.0, 1.0, 1.0],
          "refquat": [1.0, 1.0, 0.0, 0.0],
          "extra": "foo"
        }""")
        cls.missing_extra_field_json = dedent("""
        {
          "refpos": [1.0, 1.0, 1.0],
          "extra": "foo"
        }""")

    def test_valid_metadata_returns_dataclass_instance(self):
        metadata = load_object_metadata(self.valid_json, "test_object")
        assert metadata.refpos == [1.0, 1.0, 1.0]
        assert metadata.refquat == [1.0, 1.0, 0.0, 0.0]

    def test_invalid_json_raises_error(self):
        with pytest.raises(
            InvalidObjectMetadata, match=r"Couldn't decode 'test_object' metadata."
        ):
            load_object_metadata(self.invalid_json, "test_object")

    def test_metadata_missing_required_fields(self):
        with pytest.raises(InvalidObjectMetadata, match=r"missing fields: {'refquat'}"):
            load_object_metadata(self.missing_field_json, "test_object")

    def test_metadata_extra_fields(self):
        with pytest.raises(InvalidObjectMetadata, match=r"extra fields: {'extra'}"):
            load_object_metadata(self.extra_field_json, "test_object")

    def test_metadata_missing_extra_fields(self):
        match_str = r"missing fields: {'refquat'} and extra fields: {'extra'}"
        with pytest.raises(InvalidObjectMetadata, match=match_str):
            load_object_metadata(self.missing_extra_field_json, "test_object")
