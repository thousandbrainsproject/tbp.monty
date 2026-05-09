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

import numpy as np
import numpy.typing as npt
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

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

    @given(
        stored_values=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=8),
            elements=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False),
        ),
        query=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False),
        tolerance=st.floats(min_value=1e-3, max_value=10.0, allow_nan=False),
    )
    def test_numeric_evidence_is_linear_decay_of_absolute_distance(
        self,
        stored_values: npt.NDArray[np.float64],
        query: float,
        tolerance: float,
    ) -> None:
        evidence = self._calculate(
            stored=stored_values.reshape(-1, 1),
            query={"curvature": query},
            tolerances={"curvature": tolerance},
            weights={"curvature": 1.0},
            feature_order=["curvature"],
        )
        diffs = np.abs(stored_values - query)
        expected = np.clip(1.0 - diffs / tolerance, 0.0, None)
        np.testing.assert_allclose(evidence, expected, atol=1e-9, rtol=1e-9)

    @given(
        stored_hue=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        query_hue=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        tolerance=st.floats(min_value=1e-3, max_value=0.5, allow_nan=False),
    )
    def test_circular_hue_distance_wraps_around_unit_interval(
        self,
        stored_hue: float,
        query_hue: float,
        tolerance: float,
    ) -> None:
        evidence = self._calculate(
            stored=np.array([[stored_hue, 0.5, 0.5]], dtype=np.float64),
            query={"hsv": [query_hue, 0.5, 0.5]},
            tolerances={"hsv": [tolerance, 1.0, 1.0]},
            weights={"hsv": [1.0, 1.0, 1.0]},
            feature_order=["hsv"],
        )
        raw_diff = abs(stored_hue - query_hue)
        wrapped_diff = min(raw_diff, 1.0 - raw_diff)
        hue_evidence = max(0.0, 1.0 - wrapped_diff / tolerance)
        expected = (hue_evidence + 1.0 + 1.0) / 3.0
        np.testing.assert_allclose(evidence, [expected], atol=1e-9, rtol=1e-9)

    @given(
        stored_ids=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=8),
            elements=st.integers(min_value=0, max_value=5).map(float),
        ),
        query_id=st.integers(min_value=0, max_value=5).map(float),
    )
    def test_categorical_evidence_is_one_iff_equal(
        self,
        stored_ids: npt.NDArray[np.float64],
        query_id: float,
    ) -> None:
        evidence = self._calculate(
            stored=stored_ids.reshape(-1, 1),
            query={"object_id": query_id},
            tolerances={"object_id": 1.0},
            weights={"object_id": 1.0},
            feature_order=["object_id"],
        )
        expected = (stored_ids == query_id).astype(np.float64)
        np.testing.assert_array_equal(evidence, expected)

    @given(
        n_nodes=st.integers(min_value=1, max_value=8),
        curvature_query=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False),
        object_id_query=st.integers(min_value=0, max_value=5).map(float),
        curvature_tolerance=st.floats(min_value=1e-3, max_value=10.0, allow_nan=False),
        curvature_weight=st.floats(min_value=1e-3, max_value=10.0, allow_nan=False),
        object_id_weight=st.floats(min_value=1e-3, max_value=10.0, allow_nan=False),
        data=st.data(),
    )
    def test_features_combine_as_weighted_average(
        self,
        n_nodes: int,
        curvature_query: float,
        object_id_query: float,
        curvature_tolerance: float,
        curvature_weight: float,
        object_id_weight: float,
        data: st.DataObject,
    ) -> None:
        curvature_stored = data.draw(
            arrays(
                dtype=np.float64,
                shape=n_nodes,
                elements=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False),
            )
        )
        object_id_stored = data.draw(
            arrays(
                dtype=np.float64,
                shape=n_nodes,
                elements=st.integers(min_value=0, max_value=5).map(float),
            )
        )

        evidence = self._calculate(
            stored=np.stack([curvature_stored, object_id_stored], axis=1),
            query={"curvature": curvature_query, "object_id": object_id_query},
            tolerances={"curvature": curvature_tolerance, "object_id": 1.0},
            weights={"curvature": curvature_weight, "object_id": object_id_weight},
            feature_order=["curvature", "object_id"],
        )
        curvature_ev = np.clip(
            1.0 - np.abs(curvature_stored - curvature_query) / curvature_tolerance,
            0.0,
            None,
        )
        object_id_ev = (object_id_stored == object_id_query).astype(np.float64)
        expected = (
            curvature_ev * curvature_weight + object_id_ev * object_id_weight
        ) / (curvature_weight + object_id_weight)
        np.testing.assert_allclose(evidence, expected, atol=1e-9, rtol=1e-9)

    @given(
        stored_values=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=8),
            elements=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False),
        ),
        query=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False),
        tolerance=st.floats(min_value=1e-3, max_value=10.0, allow_nan=False),
        pose_vectors_value=st.floats(
            min_value=-100.0, max_value=100.0, allow_nan=False
        ),
        pose_fully_defined_value=st.floats(
            min_value=-100.0, max_value=100.0, allow_nan=False
        ),
    )
    def test_skip_features_do_not_change_result(
        self,
        stored_values: npt.NDArray[np.float64],
        query: float,
        tolerance: float,
        pose_vectors_value: float,
        pose_fully_defined_value: float,
    ) -> None:
        stored = stored_values.reshape(-1, 1)
        baseline = self._calculate(
            stored=stored,
            query={"curvature": query},
            tolerances={"curvature": tolerance},
            weights={"curvature": 1.0},
            feature_order=["curvature"],
        )
        with_skip = self._calculate(
            stored=stored,
            query={
                "pose_vectors": pose_vectors_value,
                "curvature": query,
                "pose_fully_defined": pose_fully_defined_value,
            },
            tolerances={"curvature": tolerance},
            weights={"curvature": 1.0},
            feature_order=["pose_vectors", "curvature", "pose_fully_defined"],
        )
        np.testing.assert_array_equal(baseline, with_skip)


if __name__ == "__main__":
    unittest.main()
