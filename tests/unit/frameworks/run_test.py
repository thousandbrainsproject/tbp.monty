# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from tbp.monty.frameworks.run import save_config_yaml


class SaveConfigYamlTest(unittest.TestCase):
    def test_writes_the_composed_config_unresolved(self) -> None:
        config = OmegaConf.create(
            {"experiment": {"config": {"logging": {"run_name": "${oc.env:USER}"}}}}
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = save_config_yaml(config, Path(tmp))
            self.assertEqual(path.name, "config.yaml")
            text = path.read_text()

        self.assertIn("run_name: ${oc.env:USER}", text)  # snapshot notation
        self.assertEqual(text, OmegaConf.to_yaml(config))
