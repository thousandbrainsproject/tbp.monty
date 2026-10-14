# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import logging
from unittest import TestCase
from unittest.mock import MagicMock, patch

from tbp.monty.frameworks.models.evidence_matching import graph_memory
from tbp.monty.frameworks.models.evidence_matching.graph_memory import (
    EvidenceGraphMemory,
)
from tbp.monty.frameworks.models.object_model import GridTooSmallError


class EvidenceGraphMemoryTest(TestCase):
    def setUp(self) -> None:
        self.memory = EvidenceGraphMemory(
            max_nodes_per_graph=10,
            max_graph_size=0.3,
            num_model_voxels_per_dim=50,
        )

    def test_build_graph_warns_when_grid_is_too_small(self) -> None:
        with patch.object(graph_memory, "GridObjectModel") as model_class:
            model_class.return_value.build_model.side_effect = GridTooSmallError
            with self.assertLogs(graph_memory.logger, level="WARNING") as captured:
                self.memory._build_graph(
                    locations=MagicMock(),
                    features=MagicMock(),
                    graph_id="test_object",
                    input_channel="patch",
                )

        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].levelno, logging.WARNING)
        self.assertEqual(
            captured.records[0].getMessage(),
            "Grid too small for given locations. Not building a model for test_object",
        )
        self.assertEqual(self.memory.models_in_memory, {})

    def test_build_graph_success_does_not_warn(self) -> None:
        with patch.object(graph_memory, "GridObjectModel") as model_class:
            with self.assertLogs(graph_memory.logger, level="INFO") as captured:
                self.memory._build_graph(
                    locations=MagicMock(),
                    features=MagicMock(),
                    graph_id="test_object",
                    input_channel="patch",
                )

        self.assertTrue(
            all(record.levelno < logging.WARNING for record in captured.records)
        )
        self.assertIs(
            self.memory.models_in_memory["test_object"]["patch"],
            model_class.return_value,
        )
