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
from unittest.mock import Mock

import numpy as np
import pytest
from skimage.color import rgb2gray

from tbp.monty.frameworks.models.abstract_monty_classes import SensorObservation
from tbp.monty.frameworks.models.sensor_modules import CameraSM
from tbp.monty.frameworks.utils.sensor_processing import (
    get_ltp_texture_feature_vector,
)

PATCH_SIZE = 64

LTP_CONFIG = {
    "texture_extraction": {
        "local_ternary_pattern": {
            "n_neighbors": 8,
            "radius": 1.0,
            "threshold": 5.0,
        }
    }
}

UNIFORM_LTP_CONFIG = {
    "texture_extraction": {
        "local_ternary_pattern": {
            "n_neighbors": 8,
            "radius": 1.0,
            "threshold": 5.0,
            "method": "uniform",
        }
    }
}


def _make_on_object_observation(seed: int = 0) -> SensorObservation:
    """Build a synthetic on-object distant-agent observation.

    The RGB patch is random (so the LTP histogram is non-trivial) and the point
    cloud is a slanted plane fully on the object.

    Returns:
        A SensorObservation with rgba, depth, semantic_3d, sensor_frame_data, and
        cam_to_world populated.
    """
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 256, size=(PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
    alpha = np.full((PATCH_SIZE, PATCH_SIZE, 1), 255, dtype=np.uint8)
    rgba = np.concatenate([rgb, alpha], axis=-1)
    depth = np.ones((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)

    xs, ys = np.meshgrid(
        np.linspace(-0.05, 0.05, PATCH_SIZE),
        np.linspace(-0.05, 0.05, PATCH_SIZE),
    )
    zs = 0.1 + 0.2 * xs + 0.05 * ys
    semantic_3d = np.zeros((PATCH_SIZE * PATCH_SIZE, 4), dtype=np.float64)
    semantic_3d[:, 0] = xs.ravel()
    semantic_3d[:, 1] = ys.ravel()
    semantic_3d[:, 2] = zs.ravel()
    semantic_3d[:, 3] = 1  # on object

    obs = SensorObservation(
        rgba=rgba, depth=depth, cam_to_world=np.identity(4)
    )
    obs.update(semantic_3d=semantic_3d, sensor_frame_data=semantic_3d.copy())
    return obs


def _make_partially_on_object_observation(seed: int = 0) -> SensorObservation:
    """Build an observation whose right-hand columns are off the object.

    The patch center stays on object (so the SM extracts features), but the
    rightmost quarter of pixels are background (semantic id 0). This lets us
    verify that the on-object mask actually changes the texture histogram.

    Returns:
        A SensorObservation with a partially on-object semantic map.
    """
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 256, size=(PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
    alpha = np.full((PATCH_SIZE, PATCH_SIZE, 1), 255, dtype=np.uint8)
    rgba = np.concatenate([rgb, alpha], axis=-1)
    depth = np.ones((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)

    xs, ys = np.meshgrid(
        np.linspace(-0.05, 0.05, PATCH_SIZE),
        np.linspace(-0.05, 0.05, PATCH_SIZE),
    )
    zs = 0.1 + 0.2 * xs + 0.05 * ys
    semantic_3d = np.zeros((PATCH_SIZE * PATCH_SIZE, 4), dtype=np.float64)
    semantic_3d[:, 0] = xs.ravel()
    semantic_3d[:, 1] = ys.ravel()
    semantic_3d[:, 2] = zs.ravel()

    on_object_grid = np.ones((PATCH_SIZE, PATCH_SIZE), dtype=bool)
    # Mark the rightmost quarter of columns as off object (background).
    on_object_grid[:, (3 * PATCH_SIZE) // 4 :] = False
    semantic_3d[:, 3] = on_object_grid.ravel().astype(np.float64)

    obs = SensorObservation(
        rgba=rgba, depth=depth, cam_to_world=np.identity(4)
    )
    obs.update(semantic_3d=semantic_3d, sensor_frame_data=semantic_3d.copy())
    return obs


class CameraSMLtpTest(unittest.TestCase):
    def test_ltp_config_is_required_when_ltp_feature_requested(self) -> None:
        with pytest.raises(ValueError, match="ltp_config"):
            CameraSM(
                sensor_module_id="patch",
                features=["on_object", "ltp"],
            )

    def test_ltp_config_not_required_without_ltp_feature(self) -> None:
        # Should construct without raising even though no ltp_config is given.
        CameraSM(
            sensor_module_id="patch",
            features=["on_object", "hsv"],
        )

    def test_process_extracts_ltp_histogram(self) -> None:
        sm = CameraSM(
            sensor_module_id="patch",
            features=[
                "pose_vectors",
                "pose_fully_defined",
                "on_object",
                "object_coverage",
                "hsv",
                "principal_curvatures_log",
                "ltp",
            ],
            ltp_config=LTP_CONFIG,
        )
        sm.reset()
        ctx = Mock()
        ctx.rng = np.random.RandomState(0)

        percept = sm.step(ctx, _make_on_object_observation())

        assert "ltp" in percept.non_morphological_features
        ltp = np.asarray(percept.non_morphological_features["ltp"])
        assert ltp.ndim == 1
        # n_neighbors=8 -> 36 rotation-invariant bins per sign, concatenated -> 72.
        assert ltp.shape == (72,)
        assert np.all(ltp >= 0.0)
        np.testing.assert_allclose(ltp.sum(), 1.0, atol=1e-3)

    def test_process_ltp_matches_direct_extraction(self) -> None:
        # The histogram produced by the SM matches a direct call on the grayscale
        # patch, confirming the ltp_config is plumbed through unchanged.
        obs = _make_on_object_observation(seed=3)
        sm = CameraSM(
            sensor_module_id="patch",
            features=["on_object", "ltp"],
            ltp_config=LTP_CONFIG,
        )
        sm.reset()
        ctx = Mock()
        ctx.rng = np.random.RandomState(0)
        percept = sm.step(ctx, obs)

        gray = rgb2gray(obs["rgba"][:, :, :3])
        if np.max(gray) <= 1.0:
            gray = (gray * 255.0).astype(np.uint8)
        expected = get_ltp_texture_feature_vector(gray, LTP_CONFIG)

        np.testing.assert_allclose(
            percept.non_morphological_features["ltp"], expected, rtol=0, atol=1e-9
        )

    def test_process_ltp_excludes_off_object_pixels(self) -> None:
        # The SM histograms only on-object pixels, so its output matches a
        # masked direct extraction and differs from the unmasked one.
        obs = _make_partially_on_object_observation(seed=5)
        sm = CameraSM(
            sensor_module_id="patch",
            features=["on_object", "ltp"],
            ltp_config=UNIFORM_LTP_CONFIG,
        )
        sm.reset()
        ctx = Mock()
        ctx.rng = np.random.RandomState(0)
        percept = sm.step(ctx, obs)

        gray = (rgb2gray(obs["rgba"][:, :, :3]) * 255.0).astype(np.uint8)
        mask = (obs["semantic_3d"][:, 3] > 0).reshape(gray.shape)
        expected_masked = get_ltp_texture_feature_vector(
            gray, UNIFORM_LTP_CONFIG, mask=mask
        )
        expected_unmasked = get_ltp_texture_feature_vector(gray, UNIFORM_LTP_CONFIG)

        np.testing.assert_allclose(
            percept.non_morphological_features["ltp"],
            expected_masked,
            rtol=0,
            atol=1e-9,
        )
        # The mask must actually matter: excluding the background changes the
        # histogram relative to using the whole patch.
        assert not np.allclose(expected_masked, expected_unmasked)

    def test_process_uniform_histogram_has_expected_length(self) -> None:
        # n_neighbors=8 -> p*(p-1)+3 = 59 uniform bins per sign, concatenated.
        sm = CameraSM(
            sensor_module_id="patch",
            features=["on_object", "ltp"],
            ltp_config=UNIFORM_LTP_CONFIG,
        )
        sm.reset()
        ctx = Mock()
        ctx.rng = np.random.RandomState(0)
        percept = sm.step(ctx, _make_on_object_observation())

        ltp = np.asarray(percept.non_morphological_features["ltp"])
        assert ltp.shape == (2 * (8 * 7 + 3),)
        np.testing.assert_allclose(ltp.sum(), 1.0, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
