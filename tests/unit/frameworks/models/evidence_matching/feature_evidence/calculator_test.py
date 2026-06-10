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

import cv2
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

    def test_histogram_identical_to_query_gives_full_evidence(self) -> None:
        # A node whose stored histogram equals the query histogram has chi-square
        # distance 0 and therefore receives evidence 1.
        query_hist = [0.1, 0.2, 0.3, 0.4]
        stored = np.array([query_hist], dtype=np.float64)
        evidence = self._calculate(
            stored=stored,
            query={"ltp": query_hist},
            tolerances={"ltp": 1.0},
            weights={"ltp": 1.0},
            feature_order=["ltp"],
        )
        np.testing.assert_allclose(evidence, [1.0], atol=1e-9)

    def test_histogram_evidence_matches_chi_square_decay(self) -> None:
        # Per-node evidence is clip(tol - chi, 0) / tol, where chi is cv2's
        # chi-square distance between the stored and query histograms.
        query_hist = [0.25, 0.25, 0.25, 0.25]
        stored = np.array(
            [
                [0.25, 0.25, 0.25, 0.25],  # identical -> chi 0
                [0.40, 0.20, 0.20, 0.20],  # mildly different
                [0.90, 0.05, 0.03, 0.02],  # very different
            ],
            dtype=np.float64,
        )
        tolerance = 0.5
        evidence = self._calculate(
            stored=stored,
            query={"ltp": query_hist},
            tolerances={"ltp": tolerance},
            weights={"ltp": 1.0},
            feature_order=["ltp"],
        )

        query_f32 = np.array(query_hist, dtype=np.float32)
        expected = []
        for row in stored:
            chi = cv2.compareHist(
                row.astype(np.float32), query_f32, cv2.HISTCMP_CHISQR
            )
            expected.append(max(0.0, 1.0 - chi / tolerance))
        np.testing.assert_allclose(evidence, expected, atol=1e-6)

    def test_histogram_distance_at_or_above_tolerance_gives_zero(self) -> None:
        query_hist = [1.0, 0.0, 0.0, 0.0]
        # Stored mass is entirely in a different bin -> large chi-square distance.
        stored = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
        evidence = self._calculate(
            stored=stored,
            query={"ltp": query_hist},
            tolerances={"ltp": 0.5},
            weights={"ltp": 1.0},
            feature_order=["ltp"],
        )
        np.testing.assert_allclose(evidence, [0.0], atol=1e-9)

    def test_histogram_contributes_as_single_feature_in_average(self) -> None:
        # The histogram weight is spread across its bins so that, combined with a
        # numeric feature, the result is the weight-combination of the two
        # per-feature evidences (the histogram is not counted once per bin).
        query_hist = [0.25, 0.25, 0.25, 0.25]
        stored_hist = [0.40, 0.20, 0.20, 0.20]
        curvature_stored = 2.0
        curvature_query = 2.5
        curvature_tol = 1.0
        ltp_tol = 0.5
        ltp_weight = 3.0
        curvature_weight = 1.0

        stored = np.array(
            [[curvature_stored, *stored_hist]],
            dtype=np.float64,
        )
        evidence = self._calculate(
            stored=stored,
            query={"curvature": curvature_query, "ltp": query_hist},
            tolerances={"curvature": curvature_tol, "ltp": ltp_tol},
            weights={"curvature": curvature_weight, "ltp": ltp_weight},
            feature_order=["curvature", "ltp"],
        )

        curvature_ev = max(
            0.0, 1.0 - abs(curvature_stored - curvature_query) / curvature_tol
        )
        chi = cv2.compareHist(
            np.array(stored_hist, dtype=np.float32),
            np.array(query_hist, dtype=np.float32),
            cv2.HISTCMP_CHISQR,
        )
        ltp_ev = max(0.0, 1.0 - chi / ltp_tol)
        expected = (
            curvature_ev * curvature_weight + ltp_ev * ltp_weight
        ) / (curvature_weight + ltp_weight)
        np.testing.assert_allclose(evidence, [expected], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
