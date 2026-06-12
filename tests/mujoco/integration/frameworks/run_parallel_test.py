# Copyright 2025-2026 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig

from tbp.monty.frameworks.run_parallel import main
from tests import HYDRA_ROOT


class RunParallelTest(unittest.TestCase):
    """Tests for the run_parallel functionality.

    This suite of tests verifies that running experiment episodes serially results
    in the same output as running the experiment episodes in parallel, under various
    scenarios.
    """

    output_dir: ClassVar[Path]
    supervised_pre_training_cfg: ClassVar[DictConfig]
    eval_cfg: ClassVar[DictConfig]
    eval_lt_cfg: ClassVar[DictConfig]
    eval_gt_cfg: ClassVar[DictConfig]

    @classmethod
    def setUpClass(cls) -> None:
        cls.output_dir = Path(tempfile.mkdtemp())

        cls._load_hydra_configs()

        # Run the training experiments once for this whole test suite.
        # serial run
        exp = hydra.utils.instantiate(cls.supervised_pre_training_cfg.experiment)
        with exp:
            exp.run()

        # parallel run
        main(cls.supervised_pre_training_cfg)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.output_dir)

    @classmethod
    def _load_hydra_configs(cls) -> None:
        def hydra_config(
            test_name: str, output_dir: Path, model_name_or_path: Path | None = None
        ) -> DictConfig:
            overrides = [
                f"experiment=test/{test_name}",
                "num_parallel=1",
                f"++experiment.config.logging.output_dir={output_dir}",
            ]
            if model_name_or_path:
                overrides.append(
                    f"experiment.config.model_name_or_path={model_name_or_path}"
                )
            return hydra.compose(config_name="experiment", overrides=overrides)

        with hydra.initialize_config_dir(version_base=None, config_dir=str(HYDRA_ROOT)):
            cls.supervised_pre_training_cfg = hydra_config(
                "supervised_pre_training_mujoco", cls.output_dir
            )
            cls.eval_cfg = hydra_config(
                "eval_mujoco",
                output_dir=cls.output_dir / "eval",
                model_name_or_path=cls.output_dir / "pretrained",
            )
            cls.eval_lt_cfg = hydra_config("eval_lt_mujoco", cls.output_dir / "lt")
            cls.eval_gt_cfg = hydra_config("eval_gt_mujoco", cls.output_dir / "gt")

    def test_run_parallel_equals_serial_for_training(self) -> None:
        parallel_model = torch.load(
            self.output_dir
            / self.supervised_pre_training_cfg.experiment.config.logging.run_name
            / "pretrained"
            / "model.pt",
            weights_only=False,
        )
        serial_model = torch.load(
            self.output_dir / "pretrained" / "model.pt", weights_only=False
        )

        # Same objects
        self.assertEqual(
            parallel_model["lm_dict"][0].keys(), serial_model["lm_dict"][0].keys()
        )

        # Same number of features
        self.assertEqual(
            parallel_model["lm_dict"][0]["graph_memory"]["capsule3DSolid"][
                "patch"
            ].x.size(1),
            serial_model["lm_dict"][0]["graph_memory"]["capsule3DSolid"][
                "patch"
            ].x.size(1),
        )

        # Same number of data points
        self.assertEqual(
            parallel_model["lm_dict"][0]["graph_memory"]["capsule3DSolid"][
                "patch"
            ].num_nodes,
            serial_model["lm_dict"][0]["graph_memory"]["capsule3DSolid"][
                "patch"
            ].num_nodes,
        )

    def test_eval_parallel_equals_serial_when_epochs_equals_rotations(self) -> None:
        # serial run
        exp = hydra.utils.instantiate(self.eval_cfg.experiment)
        with exp:
            exp.run()

        # parallel run
        main(self.eval_cfg)

        eval_dir = self.output_dir / "eval"
        parallel_eval_dir = eval_dir / "test_eval_mujoco"

        # Check that csv files are the same
        # Note that you can't easily do this if they actually run in parallel because
        # you don't know the execution order so it will have the same data, just
        # different order

        scsv = pd.read_csv(eval_dir / "eval_stats.csv")
        pcsv = pd.read_csv(parallel_eval_dir / "eval_stats.csv")

        # We have to drop these columns because they are not the same in the parallel
        # and serial runs. In particular, 'stepwise_performance' and
        # 'stepwise_target_object' are derived from the mapping between semantic IDs to
        # names which depend on the number of objects in the environment, and
        # environments only have one object in parallel experiments.
        drop_cols = ["time", "stepwise_performance", "stepwise_target_object"]
        scsv = scsv.drop(columns=drop_cols)
        pcsv = pcsv.drop(columns=drop_cols)

        self.assertTrue(pcsv.equals(scsv))

    def test_eval_parallel_equals_serial_when_epochs_less_than_rotations(self) -> None:
        # serial run
        exp = hydra.utils.instantiate(self.eval_lt_cfg.experiment)
        with exp:
            exp.run()

        # parallel run
        main(self.eval_lt_cfg)

        eval_dir_lt = self.output_dir / "lt"
        parallel_eval_dir_lt = eval_dir_lt / "test_eval_lt_mujoco"

        scsv_lt = pd.read_csv(eval_dir_lt / "eval_stats.csv")
        pcsv_lt = pd.read_csv(parallel_eval_dir_lt / "eval_stats.csv")

        # Remove columns that are not the same in the parallel and serial runs.
        # See comment in test_eval_parallel_equals_serial_when_epochs_equals_rotations.
        drop_cols = ["time", "stepwise_performance", "stepwise_target_object"]
        scsv_lt = scsv_lt.drop(columns=drop_cols)
        pcsv_lt = pcsv_lt.drop(columns=drop_cols)

        self.assertTrue(pcsv_lt.equals(scsv_lt))

    def test_eval_parallel_equals_serial_when_epochs_greater_than_rotations(
        self,
    ) -> None:
        # serial run
        exp = hydra.utils.instantiate(self.eval_gt_cfg.experiment)
        with exp:
            exp.run()

        # parallel run
        main(self.eval_gt_cfg)

        eval_dir_gt = self.output_dir / "gt"
        parallel_eval_dir_gt = eval_dir_gt / "test_eval_gt_mujoco"

        scsv_gt = pd.read_csv(eval_dir_gt / "eval_stats.csv")
        pcsv_gt = pd.read_csv(parallel_eval_dir_gt / "eval_stats.csv")

        # Remove columns that are not the same in the parallel and serial runs.
        # See comment in test_eval_parallel_equals_serial_when_epochs_equals_rotations.
        drop_cols = ["time", "stepwise_performance", "stepwise_target_object"]
        scsv_gt = scsv_gt.drop(columns=drop_cols)
        pcsv_gt = pcsv_gt.drop(columns=drop_cols)

        self.assertTrue(pcsv_gt.equals(scsv_gt))
