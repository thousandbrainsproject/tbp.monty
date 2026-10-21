# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import unittest
from typing import Literal

import numpy as np

from tbp.monty.cmp import Message
from tbp.monty.runtime import is_location_only_step


def _message(
    sender_type: Literal["SM", "LM"],
    process_features_in_lm: bool,
) -> Message:
    return Message(
        location=np.zeros(3),
        morphological_features={
            "pose_vectors": np.eye(3),
            "pose_fully_defined": True,
        },
        non_morphological_features={},
        confidence=1.0,
        pass_message=True,
        sender_id=f"{sender_type.lower()}_0",
        sender_type=sender_type,
        process_features_in_lm=process_features_in_lm,
    )


class IsLocationOnlyStepTest(unittest.TestCase):
    def test_ignores_lm_features(self):
        sm_location_only = _message("SM", process_features_in_lm=False)
        sm_with_features = _message("SM", process_features_in_lm=True)
        lm_with_features = _message("LM", process_features_in_lm=True)

        self.assertTrue(is_location_only_step([sm_location_only, lm_with_features]))
        self.assertFalse(is_location_only_step([sm_with_features]))
