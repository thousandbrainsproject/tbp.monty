# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import unittest

import habitat_sim

from tbp.monty.simulators.tacto import DIGIT, OMNITACT, TactoSensor


class TactoSensorTest(unittest.TestCase):
    def test_digit_config(self):
        # 1 camera, 3 LEDs
        self.assertEqual(len(DIGIT.camera), 1)
        self.assertIn("cam0", DIGIT.camera)
        self.assertEqual(len(DIGIT.lights.colors), 3)

    def test_omnitact_config(self):
        # 5 cameras, 11 LEDs
        self.assertEqual(len(OMNITACT.camera), 5)
        self.assertIn("cam0", OMNITACT.camera)
        self.assertIn("cam1", OMNITACT.camera)
        self.assertIn("cam2", OMNITACT.camera)
        self.assertIn("cam3", OMNITACT.camera)
        self.assertIn("cam4", OMNITACT.camera)
        self.assertEqual(len(OMNITACT.lights.colors), 11)

    def test_tacto_sensor(self):
        sensor = TactoSensor(
            sensor_id="sid",
            resolution=[10, 10],
            config=DIGIT,
        )
        specs = sensor.get_specs()
        self.assertEqual(len(specs), 1)
        self.assertIsInstance(specs[0], habitat_sim.sensor.CameraSensorSpec)
        self.assertAlmostEqual(specs[0].near, DIGIT.camera["cam0"].znear)
        self.assertAlmostEqual(float(specs[0].hfov), DIGIT.camera["cam0"].yfov)


if __name__ == "__main__":
    unittest.main()
