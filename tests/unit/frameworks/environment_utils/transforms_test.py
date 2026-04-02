# Copyright 2025-2026 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest

import numpy as np
import pytest

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environment_utils.transforms import GaussianBlurRGB
from tbp.monty.frameworks.models.abstract_monty_classes import (
    AgentObservations,
    Observations,
    SensorObservation,
)
from tbp.monty.frameworks.sensors import SensorID

AGENT_ID = AgentID("0")
SENSOR_ID = SensorID("0")


class GaussianBlurRGBInitTest(unittest.TestCase):
    def test_negative_kernel_size_raises(self):
        with pytest.raises(ValueError, match="kernel_size must be non-negative"):
            GaussianBlurRGB(agent_id=AGENT_ID, kernel_size=-1)

    def test_even_kernel_size_raises(self):
        with pytest.raises(ValueError, match="kernel_size must be odd or 0"):
            GaussianBlurRGB(agent_id=AGENT_ID, kernel_size=4)

    def test_non_positive_sigma_with_auto_kernel_raises(self):
        with pytest.raises(
            ValueError, match="sigma must be positive when kernel_size is 0"
        ):
            GaussianBlurRGB(agent_id=AGENT_ID, sigma=0, kernel_size=0)


class GaussianBlurRGBTest(unittest.TestCase):
    def test_sensor_id_not_in_agent(self):
        obs = Observations()
        obs[AGENT_ID] = AgentObservations()
        gaussian_smoother = GaussianBlurRGB(
            agent_id=AGENT_ID, sensor_ids=[SENSOR_ID], sigma=15, kernel_size=15
        )
        with pytest.raises(KeyError):
            gaussian_smoother(obs, _ctx=None)

    def test_rgba_not_in_sensor_observations(self):
        obs = Observations()
        obs[AGENT_ID] = AgentObservations()
        obs[AGENT_ID][SENSOR_ID] = SensorObservation()
        gaussian_smoother = GaussianBlurRGB(agent_id=AGENT_ID, sigma=15, kernel_size=15)
        with pytest.raises(KeyError):
            gaussian_smoother(obs, _ctx=None)

    def test_blur_solid_image_returns_identical(self):
        rgba_img = np.full((64, 64, 4), 128, dtype=np.uint8)
        obs = Observations()
        obs[AGENT_ID] = AgentObservations()
        obs[AGENT_ID][SENSOR_ID] = SensorObservation({"rgba": rgba_img.copy()})
        gaussian_smoother = GaussianBlurRGB(agent_id=AGENT_ID, sigma=2, kernel_size=5)
        result = gaussian_smoother(obs, _ctx=None)
        result_img = result[AGENT_ID][SENSOR_ID]["rgba"]

        self.assertEqual(result_img.shape, rgba_img.shape)
        np.testing.assert_array_equal(result_img, rgba_img)

    def test_blur_modifies_rgb_preserves_alpha(self):
        vals = np.array(
            [[1, 3, 7, 4], [5, 5, 2, 6], [4, 9, 3, 1], [2, 8, 5, 7]],
            dtype=np.float32,
        )
        rgb = np.stack([vals, vals, vals], axis=2)
        alpha = np.full((4, 4, 1), 255.0, dtype=np.float32)
        rgba_img = np.concatenate([rgb, alpha], axis=2)

        expected_img = np.array(
            [
                [4.01506871, 3.89632597, 4.42645374, 4.33937188],
                [4.83031406, 4.37522804, 4.41742389, 3.86105315],
                [6.0575654, 4.91199635, 5.0033111, 3.75015524],
                [6.69921204, 5.35835336, 5.11543164, 3.52320474],
            ]
        )

        obs = Observations()
        obs[AGENT_ID] = AgentObservations()
        obs[AGENT_ID][SENSOR_ID] = SensorObservation({"rgba": rgba_img})
        gaussian_smoother = GaussianBlurRGB(agent_id=AGENT_ID, sigma=2, kernel_size=3)
        smoothed_obs = gaussian_smoother(obs, _ctx=None)
        smoothed_img = smoothed_obs[AGENT_ID][SENSOR_ID]["rgba"]

        self.assertEqual(smoothed_img.shape, rgba_img.shape)
        expected_rgb = np.stack([expected_img] * 3, axis=2)
        np.testing.assert_allclose(smoothed_img[:, :, :3], expected_rgb, atol=1e-5)
        np.testing.assert_array_equal(smoothed_img[:, :, 3], 255.0)


if __name__ == "__main__":
    unittest.main()
