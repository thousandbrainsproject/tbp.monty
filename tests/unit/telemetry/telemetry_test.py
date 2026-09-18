# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import logging
import unittest

import hydra
import pytest

from tbp.monty import telemetry
from tbp.monty.frameworks.experiments.monty_experiment import MontyExperiment
from tbp.monty.frameworks.models import monty_base
from tbp.monty.hydra import instantiate_experiment
from tbp.monty.telemetry.publishers import TelemetryPublisher
from tbp.monty.telemetry.schemas import TelemetryEvent
from tests import HYDRA_ROOT

pytest.importorskip(
    "habitat_sim",
    reason="Habitat Sim optional dependency not installed.",
)


class TelemetryLogHandler(logging.Handler):
    """Logging handler that collects log records for telemetry assertions."""

    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord):
        self.records.append(record)


class TelemetryEventTest(unittest.TestCase):
    """Unit tests for `TelemetryEvent`."""

    def test_telemetry_event_defaults(self):
        """Verify `TelemetryEvent` kind fallback and values mapping."""
        event = TelemetryEvent(kind="")
        self.assertEqual(event.kind, event.__class__.__name__)

    def test_telemetry_event_custom_fields(self):
        """Verify `TelemetryEvent` kind when explicitly specified."""
        kind = "NewGraphAdded"
        graph_id = "new_object0"
        event = TelemetryEvent(kind=kind, graph_id=graph_id)
        self.assertEqual(event.kind, kind)
        self.assertEqual(event.graph_id, graph_id)


class TelemetryPublisherTest(unittest.TestCase):
    """Unit tests for `TelemetryPublisher`."""

    def setUp(self):
        telemetry.getTelemeter("tbp.monty").setLevel(logging.NOTSET)
        self.handler = TelemetryLogHandler()
        self.telemeter = telemetry.getTelemeter(monty_base.__name__)
        self.telemeter.setLevel(logging.NOTSET)
        self.telemeter.addHandler(self.handler)

    def tearDown(self):
        self.telemeter.removeHandler(self.handler)

    def test_params(self):
        """Verify logger params."""
        self.assertIsInstance(self.telemeter, TelemetryPublisher)
        self.assertEqual(self.telemeter.name, f"telemetry.{monty_base.__name__}")
        self.assertFalse(self.telemeter.propagate)
        self.assertTrue(self.telemeter.hasHandlers())

    def test_events(self):
        """Verify emitting events produces log records with telemetry schemas."""
        self.telemeter.setLevel(logging.DEBUG)

        events = [
            (
                self.telemeter.debug,
                logging.DEBUG,
                TelemetryEvent(kind="DebugEvent"),
            ),
            (
                self.telemeter.info,
                logging.INFO,
                TelemetryEvent(kind="InfoEvent"),
            ),
        ]

        for log_func, _, event in events:
            log_func(event)

        self.assertEqual(len(self.handler.records), len(events))

        for record, (_, level, event) in zip(self.handler.records, events):
            self.assertEqual(record.levelno, level)
            self.assertEqual(record.msg, event.kind)
            self.assertIs(record.__dict__.get("telemetry_schema"), event)

    @staticmethod
    def _instantiate_experiment(telemetry_profile: str) -> MontyExperiment:
        with hydra.initialize_config_dir(version_base=None, config_dir=str(HYDRA_ROOT)):
            base_cfg = hydra.compose(
                config_name="experiment",
                overrides=[
                    "experiment=test/profile/base",
                    f"+telemetry={telemetry_profile}",
                ],
            )
        return instantiate_experiment(base_cfg.experiment)

    def test_debug_config(self):
        """Verify behavior of Hydra config ``telemetry=debug``."""
        with self._instantiate_experiment("debug"):
            self.telemeter.debug(TelemetryEvent(kind="TestEvent"))
            self.assertEqual(self.telemeter.getEffectiveLevel(), logging.DEBUG)
            self.assertEqual(len(self.handler.records), 1)

    def test_info_config(self):
        """Verify behavior of Hydra config ``telemetry=info``."""
        with self._instantiate_experiment("info"):
            self.telemeter.debug(TelemetryEvent(kind="TestEvent"))
            self.telemeter.info(TelemetryEvent(kind="TestEvent"))
            self.assertEqual(self.telemeter.getEffectiveLevel(), logging.INFO)
            self.assertEqual(len(self.handler.records), 1)

    def test_none_config(self):
        """Verify behavior of Hydra config ``telemetry=none``."""
        with self._instantiate_experiment("none"):
            self.telemeter.info(TelemetryEvent(kind="TestEvent"))
            self.assertEqual(len(self.handler.records), 0)
