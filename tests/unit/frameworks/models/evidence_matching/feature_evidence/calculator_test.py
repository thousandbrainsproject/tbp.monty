# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import unittest

import numpy as np
import numpy.typing as npt

from tbp.monty.frameworks.models.evidence_matching.feature_evidence.calculator import (
    DefaultFeatureEvidenceCalculator,
)


class DefaultFeatureEvidenceCalculatorTest(unittest.TestCase):
    @staticmethod
    def _calculate(
        stored: npt.NDArray[np.float64],
        query: dict[str, float | list[float]],
        tolerances: dict[str, float | list[float]],
        weights: dict[str, float | list[float]],
        feature_order: list[str],
    ) -> npt.NDArray[np.float64]:
        return DefaultFeatureEvidenceCalculator.calculate(
            channel_feature_array=stored,
            channel_feature_order=feature_order,
            channel_feature_weights=weights,
            channel_query_features=query,
            channel_tolerances=tolerances,
            input_channel="patch_0",
        )

    def test_numeric_match_and_decay(self) -> None:
        evidence = self._calculate(
            stored=np.array([[0.0], [0.05], [0.10], [0.20]], dtype=np.float64),
            query={"curvature": 0.0},
            tolerances={"curvature": 0.10},
            weights={"curvature": 1.0},
            feature_order=["curvature"],
        )
        np.testing.assert_allclose(evidence, [1.0, 0.5, 0.0, 0.0])

    def test_circular_hsv_wraps_around(self) -> None:
        evidence = self._calculate(
            stored=np.array([[0.95, 0.5, 0.5]], dtype=np.float64),
            query={"hsv": [0.05, 0.5, 0.5]},
            tolerances={"hsv": [0.2, 0.2, 0.2]},
            weights={"hsv": [1.0, 1.0, 1.0]},
            feature_order=["hsv"],
        )
        # Hue distance wraps to 0.10 (not 0.90), which is half the tolerance.
        # S and V match exactly.
        np.testing.assert_allclose(evidence, [(0.5 + 1.0 + 1.0) / 3.0])

    def test_categorical_object_id_is_zero_or_one(self) -> None:
        evidence = self._calculate(
            stored=np.array([[329.0], [329.0], [330.0], [9999.0]], dtype=np.float64),
            query={"object_id": 329.0},
            tolerances={"object_id": 1.0},
            weights={"object_id": 1.0},
            feature_order=["object_id"],
        )
        np.testing.assert_array_equal(evidence, [1.0, 1.0, 0.0, 0.0])

    def test_mixed_numeric_and_categorical(self) -> None:
        evidence = self._calculate(
            stored=np.array(
                [
                    [0.0, 329.0],
                    [0.0, 9999.0],
                    [0.10, 329.0],
                ],
                dtype=np.float64,
            ),
            query={"curvature": 0.0, "object_id": 329.0},
            tolerances={"curvature": 0.10, "object_id": 1.0},
            weights={"curvature": 1.0, "object_id": 3.0},
            feature_order=["curvature", "object_id"],
        )

        #  row 0: [1.0, 1.0] -> 1.0
        #  row 1: [1.0, 0.0] -> 0.25 (weightes are [1.0, 3.0])
        #  row 2: [0.0, 1.0] -> 0.75 (weightes are [1.0, 3.0])
        np.testing.assert_allclose(evidence, [1.0, 0.25, 0.75])

    def test_skip_features_are_ignored(self) -> None:
        # pose_vectors / pose_fully_defined are in _SKIP_FEATURES, so the
        # calculator never reads their values.
        evidence = self._calculate(
            stored=np.array([[0.0, 329.0]], dtype=np.float64),
            query={
                "curvature": 0.0,
                "object_id": 329.0,
                "pose_vectors": 0.0,
                "pose_fully_defined": 0.0,
            },
            tolerances={"curvature": 0.10, "object_id": 1.0},
            weights={"curvature": 1.0, "object_id": 1.0},
            feature_order=[
                "pose_vectors",
                "curvature",
                "pose_fully_defined",
                "object_id",
            ],
        )
        np.testing.assert_allclose(evidence, [1.0])


if __name__ == "__main__":
    unittest.main()
