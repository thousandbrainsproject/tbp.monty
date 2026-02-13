# Copyright 2025-2026 Thousand Brains Project
# Copyright 2024 Numenta Inc.
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

from tbp.monty.frameworks.models.evidence_sdr_matching import (
    EncoderSDR,
    EvidenceSDRTargetOverlaps,
)


def set_seed(seed):
    """Set seed for reproducibility."""
    np.random.seed(seed)


class EvidenceSDRUnitTest(unittest.TestCase):
    def setUp(self):
        """Setup function at the beginning of each experiment."""
        # set seed for reproducibility
        set_seed(42)

    def test_unit_sdr_conf(self):
        """This tests edge condition of SDR configuration with invalid sparsity.

        For example, an SDR size of 100 with 100 on bits or 0 on bits.

        Expected behaviour:
            - Create warning
            - Adjust sparsity to 2%

        """
        encoder_sdr = EncoderSDR(
            sdr_length=100, sdr_on_bits=100, lr=1e-2, n_epochs=100, log_flag=False
        )

        # test that the sparsity is adjusted to 2%
        self.assertEqual(encoder_sdr.sdr_on_bits, int(100 * 0.02))

        encoder_sdr = EncoderSDR(
            sdr_length=100, sdr_on_bits=0, lr=1e-2, n_epochs=100, log_flag=False
        )

        # test that the sparsity is adjusted to 2%
        self.assertEqual(encoder_sdr.sdr_on_bits, int(100 * 0.02))

    def test_overlap_running_average(self):
        """This tests the model's ability to calculate the running average.

        For example, after giving the `EvidenceSDRTargetOverlaps` class
        three overlap tensors, it should store equally weighted average.

        Expected behaviour:
            - Expand to the specified number of objects
            - Track the correct running average after each overlap tensor

        """
        # define data
        first_overlaps = np.array([[1.0, 3.0], [2.0, 4.0]])
        second_overlaps = np.array([[3.0, 4.0], [8.0, 1.0]])
        third_overlaps = np.array([[6.0, 2.0], [10.0, 2.0]])

        first_second_avg = np.round((first_overlaps + second_overlaps) / 2)
        first_second_third_avg = np.round(
            (first_overlaps + second_overlaps + third_overlaps) / 3
        )

        # Instantiate target overlap class to keep track of running average
        target_overlaps = EvidenceSDRTargetOverlaps()

        # test expanding overlaps to the right size
        target_overlaps.add_objects(2)
        self.assertEqual(target_overlaps.overlaps.shape, ((2, 2)))

        # test adding first overlap to the array
        target_overlaps.add_overlaps(first_overlaps)
        self.assertTrue(np.all(first_overlaps == target_overlaps.overlaps))

        # test running average after adding second overlaps tensor
        target_overlaps.add_overlaps(second_overlaps)
        self.assertTrue(np.all(first_second_avg == target_overlaps.overlaps))

        # test running average after adding third overlaps tensor
        target_overlaps.add_overlaps(third_overlaps)
        self.assertTrue(np.all(first_second_third_avg == target_overlaps.overlaps))

    def test_unit_train_zero_objects(self):
        """This tests edge condition of not sending any data to the training function.

        For example, we add 5 objects but attempt to train with no overlap targets.

        Expected behaviour:
            - Create warning
            - Return without training

        """
        encoder = EncoderSDR(
            sdr_length=100, sdr_on_bits=5, lr=1e-2, n_epochs=100, log_flag=True
        )
        encoder.add_objects(5)

        training_data = np.full((0, 0), np.nan)
        stats = encoder.train_sdrs(training_data)

        # test that it should return without training
        self.assertTrue("training" not in stats)

    def test_unit_can_train_subset_objects(self):
        """Test training on a subset of objects.

        This tests sending overlap targets for a subset of the objects and training
        the SDRs for them.

        For example adding 5 objects and training overlaps for 3 only.

        Expected behaviour:
            - Train normally without warning
            - Representations of the first 3 objects should change
            - Representations of the last 2 objects shouldn't change
        """
        encoder = EncoderSDR(
            sdr_length=100, sdr_on_bits=5, lr=1e-2, n_epochs=100, log_flag=True
        )
        encoder.add_objects(5)
        training_data = np.full((5, 5), np.nan)
        training_data[0, 1] = 12
        training_data[0, 2] = 33
        training_data[1, 2] = 17

        stats = encoder.train_sdrs(training_data)

        # test that it should train normally
        self.assertTrue("training" in stats)

        representations_before = stats["training"][0]["obj_dense"]
        representations_after = stats["training"][90]["obj_dense"]

        # test that the first 3 representations changed
        self.assertFalse(
            np.all(representations_before[:3] == representations_after[:3])
        )

        # test that the last 2 representations did not change
        self.assertTrue(
            np.all(representations_before[-2:] == representations_after[-2:])
        )

    def test_unit_can_train_superset_objects(self):
        """Test training on a superset of objects.

        This tests sending overlap targets for a objects not added to
        the encoder and training the SDRs for them.

        For example adding 3 objects and training overlaps for 5 objects.

        Expected behaviour when some points are out of bounds:
            - Create warning
            - Remove the out of bounds keys
            - Continue to train normally for valid points

        """
        # test for some points out of bound
        encoder = EncoderSDR(
            sdr_length=100, sdr_on_bits=5, lr=1e-2, n_epochs=100, log_flag=True
        )
        encoder.add_objects(3)
        training_data = np.full((5, 5), np.nan)
        training_data[0, 1] = 12
        training_data[0, 4] = 33
        training_data[1, 4] = 17

        # some points out of bounds
        stats = encoder.train_sdrs(training_data)

        # check that target overlap only has 3 objects, not 5
        self.assertEqual(stats["target_overlap"].shape, ((3, 3)))

        # check that the value at key 0,1 is 12
        self.assertEqual(stats["target_overlap"][0, 1], 12)

        # test for all points out of bounds
        encoder = EncoderSDR(
            sdr_length=100, sdr_on_bits=5, lr=1e-2, n_epochs=100, log_flag=True
        )
        encoder.add_objects(3)
        training_data = np.full((5, 5), np.nan)
        training_data[0, 3] = 12
        training_data[0, 4] = 33
        training_data[1, 4] = 17

        stats = encoder.train_sdrs(training_data)

        # check that the value of all target overlaps are 0.0
        self.assertTrue(np.all(stats["target_overlap"] == 0.0))

    def test_unit_can_minimize_train_error(self):
        """This tests the model's ability to follow target overlap.

        For example, the error in training 3 objects should be small.

        Expected behaviour:
            - Train normally without warning
            - Overlap error should be decreasing over time during training
            - SDR overlaps should be close to target overlaps
        """
        encoder = EncoderSDR(
            sdr_length=2048, sdr_on_bits=41, lr=1e-2, n_epochs=1000, log_flag=True
        )
        encoder.add_objects(3)
        training_data = np.full((3, 3), np.nan)
        training_data[0, 1] = 12
        training_data[0, 2] = 33
        training_data[1, 2] = 17

        stats = encoder.train_sdrs(training_data)

        mask = stats["mask"]
        overlap_error_before = stats["training"][0]["overlap_error"][mask == 1].mean()
        overlap_error_after = stats["training"][990]["overlap_error"][mask == 1].mean()

        condition_a = overlap_error_after < overlap_error_before  # error decreasing
        condition_b = overlap_error_after <= 2.0  # small average overalp error
        self.assertTrue(condition_a and condition_b)

    def test_can_stabilize_sdrs(self):
        """Test ability to stabilize old object SDRs when new objects are added.

        This behavior is controlled through the `stability` argument passed to the
        `sdr_args` in the LM args.

        In this unit test, we train 3 SDRs, then train a 4th SDR. Therefore,
        the stability will be applied on the 3 trained SDRs. We save a copy
        of those 3 SDRs before training the 4th SDR (before_sdrs), and compare
        it to the 3 SDRs after training the 4th SDR (after_sdrs). Depending on
        the amount of stability applied, they should be very similar
        (high stability) or not similar (low stability).

        We matrix multiply the SDRs (before and after) such that the overlaps
        will be on the diagonal. If the SDRs didn't change, we get the highest
        overlap (i.e., 41). But if they changed we get less that 41 overlap
        bits.

        Here we will test multiple stability values and test if the SDRs
        are more stable with higher stability values.

        Expected behaviour:
            - Train normally without warning
            - Stability value more than 0 stabilizes the SDRs
            - Higher stability values forces SDRs to be more stable
            - Stability value of 1 fixes SDRs to original value
        """
        sdr_persistence = []
        for stability in [0.0, 0.33, 0.66, 1.0]:
            set_seed(42)

            encoder = EncoderSDR(
                sdr_length=2048,
                sdr_on_bits=41,
                lr=1e-2,
                n_epochs=1000,
                stability=stability,
                log_flag=True,
            )

            encoder.add_objects(3)
            training_data = np.full((3, 3), np.nan)
            training_data[0, 1] = 12
            training_data[0, 2] = 33
            training_data[1, 2] = 17
            encoder.train_sdrs(training_data)
            before_sdrs = encoder.sdrs.copy()

            encoder.add_objects(1)
            training_data = np.full((4, 4), np.nan)
            training_data[0, 1] = 12
            training_data[0, 2] = 33
            training_data[1, 2] = 17
            training_data[0, 3] = 38
            training_data[1, 3] = 13
            training_data[2, 3] = 30
            encoder.train_sdrs(training_data)
            after_sdrs = encoder.sdrs.copy()[:3]

            # compare the older sdrs before and after training with new objects
            sdr_persistence.append(np.mean(np.diag(before_sdrs @ after_sdrs.T)))

        # test increasing stability
        for i in range(len(sdr_persistence) - 1):
            self.assertTrue(sdr_persistence[i + 1] > sdr_persistence[i])

        # test fixed sdrs at stability = 1.0
        self.assertEqual(sdr_persistence[-1], 41.0)

    def test_unit_can_train_identical_sdrs(self):
        """This tests the model's ability to train identical sdrs.

        Expected behaviour:
            - Train normally without warning
            - SDRs should be identical with overlap of exactly `sdr_on_bits`
        """
        encoder = EncoderSDR(
            sdr_length=2048, sdr_on_bits=41, lr=1e-2, n_epochs=1000, log_flag=True
        )
        encoder.add_objects(2)
        training_data = np.full((2, 2), np.nan)
        training_data[0, 1] = 41

        # object representations should not be identical at first
        sdrs = encoder.sdrs.copy()
        self.assertNotEqual((sdrs[0] * sdrs[1]).sum(), 41.0)

        encoder.train_sdrs(training_data)

        # object representations should become identical after training
        sdrs = encoder.sdrs.copy()
        self.assertEqual((sdrs[0] * sdrs[1]).sum(), 41.0)
