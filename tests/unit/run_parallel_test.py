# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import pytest

pytest.importorskip(
    "habitat_sim",
    reason="Habitat Sim optional dependency not installed.",
)

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from tbp.monty.frameworks.run_parallel import main


class RunParallelTest(unittest.TestCase):
    def setUp(self):
        self.output_dir = Path(tempfile.mkdtemp())

        def hydra_config(
            test_name: str, output_dir: Path, model_name_or_path: Path | None = None
        ) -> DictConfig:
            overrides = [
                f"experiment=test/{test_name}",
                "num_parallel=1",
                f"++experiment.config.logging.output_dir={output_dir}",
                "+experiment.config.monty_config.motor_system_config"
                ".motor_system_args.policy_args.file_name="
                f"{Path(__file__).parent / 'resources/fixed_test_actions.jsonl'}",
            ]
            if model_name_or_path:
                overrides.append(
                    f"experiment.config.model_name_or_path={model_name_or_path}"
                )
            return hydra.compose(config_name="experiment", overrides=overrides)

        with hydra.initialize(version_base=None, config_path="../../conf"):
            self.supervised_pre_training_cfg = hydra_config(
                "supervised_pre_training", self.output_dir
            )
            self.eval_cfg = hydra_config(
                "eval",
                output_dir=self.output_dir / "eval",
                model_name_or_path=self.output_dir / "pretrained",
            )
            self.eval_lt_cfg = hydra_config("eval_lt", self.output_dir / "lt")
            self.eval_gt_cfg = hydra_config("eval_gt", self.output_dir / "gt")

    def check_reproducibility_logs(self, serial_repro_dir, parallel_repro_dir):
        s_param_files = sorted(serial_repro_dir.glob("*target*"))
        p_param_files = sorted(parallel_repro_dir.glob("*target*"))

        # Same param files for each episode. No more, no less.
        self.assertEqual(
            {p.name for p in s_param_files}, {p.name for p in p_param_files}
        )
        for sfile, pfile in zip(s_param_files, p_param_files):
            with pfile.open() as f:
                ptarget = f.read()

            with sfile.open() as f:
                starget = f.read()

            ptarget = json.loads(ptarget)
            starget = json.loads(starget)

            pkeys = set(ptarget.keys())
            skeys = set(starget.keys())
            self.assertEqual(pkeys, skeys)

            for key in pkeys:
                self.assertEqual(ptarget[key], starget[key])

    def test_run_parallel_equals_serial_for_various_n_eval_epochs(self):
        # serial run
        exp = hydra.utils.instantiate(self.supervised_pre_training_cfg.experiment)
        with exp:
            exp.model.set_experiment_mode("train")
            exp.train()

        # parallel run
        OmegaConf.clear_resolvers()  # main will re-register resolvers
        main(self.supervised_pre_training_cfg)

        ###
        # Compare results
        ###
        parallel_model = torch.load(
            self.output_dir
            / self.supervised_pre_training_cfg.experiment.config.logging.run_name
            / "pretrained"
            / "model.pt"
        )
        serial_model = torch.load(self.output_dir / "pretrained" / "model.pt")

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

        ###
        # n_eval_epochs == len(eval_rotations)
        ###
        # serial run
        exp = hydra.utils.instantiate(self.eval_cfg.experiment)
        with exp:
            exp.evaluate()

        # parallel run
        OmegaConf.clear_resolvers()  # main will re-register resolvers
        main(self.eval_cfg)

        ###
        # Compare results
        ###
        eval_dir = self.output_dir / "eval"
        parallel_eval_dir = eval_dir / "test_eval"
        serial_repro_dir = eval_dir / "reproduce_episode_data"
        parallel_repro_dir = parallel_eval_dir / "reproduce_episode_data"

        # Check that reproducibility logger has same files for both
        self.check_reproducibility_logs(serial_repro_dir, parallel_repro_dir)

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

        ###
        # n_eval_epochs < len(eval_rotations)
        ###
        # serial run
        exp = hydra.utils.instantiate(self.eval_lt_cfg.experiment)
        with exp:
            exp.evaluate()

        # parallel run
        OmegaConf.clear_resolvers()  # main will re-register resolvers
        main(self.eval_lt_cfg)

        eval_dir_lt = self.output_dir / "lt"
        parallel_eval_dir_lt = eval_dir_lt / "test_eval_lt"
        serial_repro_dir_lt = eval_dir_lt / "reproduce_episode_data"
        parallel_repro_dir_lt = parallel_eval_dir_lt / "reproduce_episode_data"

        # Check that reproducibility logger has same files for both
        self.check_reproducibility_logs(serial_repro_dir_lt, parallel_repro_dir_lt)

        scsv_lt = pd.read_csv(eval_dir_lt / "eval_stats.csv")
        pcsv_lt = pd.read_csv(parallel_eval_dir_lt / "eval_stats.csv")

        # Remove columns that are not the same in the parallel and serial runs.
        scsv_lt = scsv_lt.drop(columns=drop_cols)
        pcsv_lt = pcsv_lt.drop(columns=drop_cols)

        self.assertTrue(pcsv_lt.equals(scsv_lt))

        ###
        # n_eval_epochs > len(eval_rotations)
        ###
        # serial run
        exp = hydra.utils.instantiate(self.eval_gt_cfg.experiment)
        with exp:
            exp.evaluate()

        # parallel run
        OmegaConf.clear_resolvers()  # main will re-register resolvers
        main(self.eval_gt_cfg)

        eval_dir_gt = self.output_dir / "gt"
        parallel_eval_dir_gt = eval_dir_gt / "test_eval_gt"
        serial_repro_dir_gt = eval_dir_gt / "reproduce_episode_data"
        parallel_repro_dir_gt = parallel_eval_dir_gt / "reproduce_episode_data"

        # Check that reproducibility logger has same files for both
        self.check_reproducibility_logs(serial_repro_dir_gt, parallel_repro_dir_gt)

        scsv_gt = pd.read_csv(eval_dir_gt / "eval_stats.csv")
        pcsv_gt = pd.read_csv(parallel_eval_dir_gt / "eval_stats.csv")

        scsv_gt = scsv_gt.drop(columns=drop_cols)
        pcsv_gt = pcsv_gt.drop(columns=drop_cols)

        self.assertTrue(pcsv_gt.equals(scsv_gt))

        shutil.rmtree(self.output_dir)


if __name__ == "__main__":
    unittest.main()
