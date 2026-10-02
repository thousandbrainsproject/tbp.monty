# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from unittest import TestCase

from tbp.monty.frameworks.models.evidence_matching.channels import (
    PoseKind,
    channel_pose_kinds,
)


class ChannelPoseKindsTest(TestCase):
    def test_maps_sensor_modules_to_surface_and_learning_modules_to_object(
        self,
    ) -> None:
        pose_kinds = channel_pose_kinds(
            {"patch_0": "SM", "patch_1": "SM", "learning_module_0": "LM"}
        )

        self.assertEqual(
            pose_kinds,
            {
                "patch_0": PoseKind.SURFACE,
                "patch_1": PoseKind.SURFACE,
                "learning_module_0": PoseKind.OBJECT,
            },
        )

    def test_no_channels_gives_empty_mapping(self) -> None:
        self.assertEqual(channel_pose_kinds({}), {})

    def test_unknown_sender_type_raises(self) -> None:
        with self.assertRaises(KeyError):
            channel_pose_kinds({"patch_0": "TwoDSM"})