# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import pytest

pytest.importorskip(
    "habitat_sim",
    reason="Habitat Sim optional dependency not installed.",
)

import copy
import json
import os
import shutil
import tempfile
import unittest
from pprint import pprint

import pandas as pd
import torch

from tbp.monty.frameworks.config_utils.config_args import (
    LoggingConfig,
    MontyArgs,
    PatchAndViewMontyConfig,
    PretrainLoggingConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataLoaderPerObjectEvalArgs,
    EnvironmentDataLoaderPerObjectTrainArgs,
    ExperimentArgs,
    PredefinedObjectInitializer,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import (
    MontyObjectRecognitionExperiment,
    MontySupervisedObjectPretrainingExperiment,
)
from tbp.monty.frameworks.models.displacement_matching import DisplacementGraphLM
from tbp.monty.frameworks.run_parallel import main as run_parallel
from tbp.monty.simulators.habitat.configs import (
    EnvInitArgsPatchViewMount,
    PatchViewFinderMountHabitatDatasetArgs,
)
from tests.unit.graph_learning_test import MotorSystemConfigFixed


class RunParallelTest(unittest.TestCase):
    def setUp(self):
        self.output_dir = tempfile.mkdtemp()
        self.train_rotations = [[0.0, 0.0, 0.0], [0.0, 45.0, 0.0], [0.0, 135.0, 0.0]]
        self.eval_rotations = [[0.0, 30.0, 0.0], [0.0, 60.0, 0.0], [0.0, 90.0, 0.0]]
        self.supervised_pre_training = dict(
            experiment_class=MontySupervisedObjectPretrainingExperiment,
            experiment_args=ExperimentArgs(
                do_eval=False,
                n_train_epochs=len(self.train_rotations),
            ),
            logging_config=PretrainLoggingConfig(output_dir=self.output_dir),
            monty_config=PatchAndViewMontyConfig(
                motor_system_config=MotorSystemConfigFixed(),
                monty_args=MontyArgs(num_exploratory_steps=10),
                learning_module_configs=dict(
                    learning_module_0=dict(
                        learning_module_class=DisplacementGraphLM,
                        learning_module_args=dict(
                            k=5,
                            match_attribute="displacement",
                        ),
                    )
                ),
            ),
            dataset_args=PatchViewFinderMountHabitatDatasetArgs(
                env_init_args=EnvInitArgsPatchViewMount(data_path=None).__dict__,
            ),
            train_dataloader_class=ED.InformedEnvironmentDataLoader,
            train_dataloader_args=EnvironmentDataLoaderPerObjectTrainArgs(
                object_names=["capsule3DSolid", "cubeSolid"],
                object_init_sampler=PredefinedObjectInitializer(
                    rotations=self.train_rotations
                ),
            ),
            eval_dataloader_class=ED.InformedEnvironmentDataLoader,
            eval_dataloader_args=EnvironmentDataLoaderPerObjectEvalArgs(
                object_names=[],
                object_init_sampler=PredefinedObjectInitializer(),
            ),
        )

        self.eval_config = copy.deepcopy(self.supervised_pre_training)
        self.eval_config.update(
            experiment_class=MontyObjectRecognitionExperiment,
            logging_config=LoggingConfig(
                output_dir=os.path.join(self.output_dir, "eval")
            ),
            experiment_args=ExperimentArgs(
                do_eval=True,
                do_train=False,
                # Tests the usual case: n_eval_epochs = len(eval_rotations)
                n_eval_epochs=len(self.eval_rotations),
                model_name_or_path=os.path.join(self.output_dir, "pretrained"),
            ),
            eval_dataloader_args=EnvironmentDataLoaderPerObjectEvalArgs(
                object_names=["capsule3DSolid", "cubeSolid"],
                object_init_sampler=PredefinedObjectInitializer(
                    rotations=self.eval_rotations
                ),
            ),
            monty_config=PatchAndViewMontyConfig(
                motor_system_config=MotorSystemConfigFixed(),
                monty_args=MontyArgs(num_exploratory_steps=10),
                learning_module_configs=dict(
                    learning_module_0=dict(
                        learning_module_class=DisplacementGraphLM,
                        learning_module_args=dict(
                            k=5,
                            match_attribute="displacement",
                        ),
                    )
                ),
            ),
        )

        # This tests that parallel code can handle n_eval_epochs < len(rotations)
        self.eval_config_lt = copy.deepcopy(self.eval_config)
        self.eval_config_lt["experiment_args"].n_eval_epochs = 2
        self.output_dir_lt = os.path.join(self.output_dir, "lt")
        self.eval_config_lt["logging_config"].output_dir = self.output_dir_lt
        os.makedirs(self.eval_config_lt["logging_config"].output_dir)

        # This tests that parallel code can handle n_eval_epochs > len(rotations)
        self.eval_config_gt = copy.deepcopy(self.eval_config)
        self.eval_config_gt["experiment_args"].n_eval_epochs = 4
        self.output_dir_gt = os.path.join(self.output_dir, "gt")
        self.eval_config_gt["logging_config"].output_dir = self.output_dir_gt
        os.makedirs(self.eval_config_gt["logging_config"].output_dir)

    def check_reproducibility_logs(self, serial_repro_dir, parallel_repro_dir):
        s_param_files = [i for i in os.listdir(serial_repro_dir) if "target" in i]
        p_param_files = [i for i in os.listdir(parallel_repro_dir) if "target" in i]

        # Same param files for each episode. No more, no less.
        self.assertEqual(set(s_param_files), set(p_param_files))
        for file in s_param_files:
            pfile = os.path.join(parallel_repro_dir, file)
            sfile = os.path.join(serial_repro_dir, file)

            with open(pfile) as f:
                ptarget = f.read()

            with open(sfile) as f:
                starget = f.read()

            ptarget = json.loads(ptarget)
            starget = json.loads(starget)

            pkeys = set(ptarget.keys())
            skeys = set(starget.keys())
            self.assertEqual(pkeys, skeys)

            for key in pkeys:
                self.assertEqual(ptarget[key], starget[key])

    def test_parallel_runs_n_epochs_lt(self):
        #####
        # Train
        #####

        ###
        # Run training like normal in serial
        ###
        pprint("...Setting up serial experiment...")
        config = self.supervised_pre_training
        with MontySupervisedObjectPretrainingExperiment(config) as exp:
            exp.model.set_experiment_mode("train")

            pprint("...Training in serial...")
            exp.train()

        ###
        # Run training with run_parallel
        ###
        pprint("...Setting up parallel experiment...")
        run_parallel(
            exp=self.supervised_pre_training,
            experiment="unittest_supervised_pre_training",
            num_parallel=1,
            quiet_habitat_logs=True,
            print_cfg=False,
            is_unittest=True,
        )

        ###
        # Compare results
        ###
        parallel_model = torch.load(
            os.path.join(
                self.output_dir,
                "unittest_supervised_pre_training",
                "pretrained",
                "model.pt",
            )
        )
        serial_model = torch.load(
            os.path.join(self.output_dir, "pretrained", "model.pt")
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

        #####
        # Testing
        #####

        ###
        # n_eval_epochs = len(rotations)
        ###

        # In serial like normal
        pprint("...Setting up serial experiment...")
        with MontyObjectRecognitionExperiment(self.eval_config) as eval_exp:
            pprint("...Evaluating in serial...")
            eval_exp.evaluate()

        # Using run_parallel
        pprint("...Setting up parallel experiment...")
        run_parallel(
            exp=self.eval_config,
            experiment="unittest_eval_eq",
            num_parallel=1,
            quiet_habitat_logs=True,
            print_cfg=False,
            is_unittest=True,
        )

        eval_dir = os.path.join(self.output_dir, "eval")
        parallel_eval_dir = os.path.join(eval_dir, "unittest_eval_eq")
        serial_repro_dir = os.path.join(eval_dir, "reproduce_episode_data")
        parallel_repro_dir = os.path.join(parallel_eval_dir, "reproduce_episode_data")

        # Check that reproducibility logger has same files for both
        self.check_reproducibility_logs(serial_repro_dir, parallel_repro_dir)

        # Check that csv files are the same
        # Note that you can't easily do this if they actually run in parallel because
        # you don't know the execution order so it will have the same data, just
        # different order

        scsv = pd.read_csv(os.path.join(eval_dir, "eval_stats.csv"))
        pcsv = pd.read_csv(os.path.join(parallel_eval_dir, "eval_stats.csv"))

        # We have to drop these columns because they are not the same in the parallel
        # and serial runs. In particular, 'stepwise_performance' and
        # 'stepwise_target_object' are derived from the mapping between semantic IDs to
        #  names which depend on the number of objects in the data loader, and data
        # loaders only have one object in parallel experiments.
        for col in ["time", "stepwise_performance", "stepwise_target_object"]:
            scsv.drop(columns=col, inplace=True)
            pcsv.drop(columns=col, inplace=True)

        self.assertTrue(pcsv.equals(scsv))

        ###
        # n_eval_epochs < len(rotations)
        ###

        # In serial like normal
        pprint("...Setting up serial experiment...")
        with MontyObjectRecognitionExperiment(self.eval_config_lt) as eval_exp_lt:
            pprint("...Evaluating in serial...")
            eval_exp_lt.evaluate()

        # Using run_parallel
        pprint("...Setting up parallel experiment...")
        run_parallel(
            exp=self.eval_config_lt,
            experiment="unittest_eval_lt",
            num_parallel=1,
            quiet_habitat_logs=True,
            print_cfg=False,
            is_unittest=True,
        )

        eval_dir_lt = self.output_dir_lt
        parallel_eval_dir_lt = os.path.join(eval_dir_lt, "unittest_eval_lt")
        serial_repro_dir_lt = os.path.join(eval_dir_lt, "reproduce_episode_data")
        parallel_repro_dir_lt = os.path.join(
            parallel_eval_dir_lt, "reproduce_episode_data"
        )

        # Check that reproducibility logger has same files for both
        self.check_reproducibility_logs(serial_repro_dir_lt, parallel_repro_dir_lt)

        scsv_lt = pd.read_csv(os.path.join(eval_dir_lt, "eval_stats.csv"))
        pcsv_lt = pd.read_csv(os.path.join(parallel_eval_dir_lt, "eval_stats.csv"))

        # Remove columns that are not the same in the parallel and serial runs.
        for col in ["time", "stepwise_performance", "stepwise_target_object"]:
            scsv_lt.drop(columns=col, inplace=True)
            pcsv_lt.drop(columns=col, inplace=True)

        self.assertTrue(pcsv_lt.equals(scsv_lt))

        # In serial like normal
        pprint("...Setting up serial experiment...")
        with MontyObjectRecognitionExperiment(self.eval_config_gt) as eval_exp_gt:
            pprint("...Evaluating in serial...")
            eval_exp_gt.evaluate()

        # Using run_parallel
        pprint("...Setting up parallel experiment...")
        run_parallel(
            exp=self.eval_config_gt,
            experiment="unittest_eval_gt",
            num_parallel=1,
            quiet_habitat_logs=True,
            print_cfg=False,
            is_unittest=True,
        )

        eval_dir_gt = self.output_dir_gt
        parallel_eval_dir_gt = os.path.join(eval_dir_gt, "unittest_eval_gt")
        serial_repro_dir_gt = os.path.join(eval_dir_gt, "reproduce_episode_data")
        parallel_repro_dir_gt = os.path.join(
            parallel_eval_dir_gt, "reproduce_episode_data"
        )

        # Check that reproducibility logger has same files for both
        self.check_reproducibility_logs(serial_repro_dir_gt, parallel_repro_dir_gt)

        scsv_gt = pd.read_csv(os.path.join(eval_dir_gt, "eval_stats.csv"))
        pcsv_gt = pd.read_csv(os.path.join(parallel_eval_dir_gt, "eval_stats.csv"))

        for col in ["time", "stepwise_performance", "stepwise_target_object"]:
            scsv_gt.drop(columns=col, inplace=True)
            pcsv_gt.drop(columns=col, inplace=True)

        self.assertTrue(pcsv_gt.equals(scsv_gt))

        shutil.rmtree(self.output_dir)


if __name__ == "__main__":
    unittest.main()
