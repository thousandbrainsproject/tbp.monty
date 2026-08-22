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
import shutil
import tempfile
import time
import unittest

import hydra
import pytest

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks import telemetry
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models import monty_base
from tbp.monty.frameworks.telemetry.publishers import TelemetryPublisher
from tbp.monty.frameworks.telemetry.schemas import TelemetryEvent, TelemetrySchema
from tbp.monty.hydra import instantiate_experiment
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


class TelemetrySchemaTest(unittest.TestCase):
    """Unit tests for telemetry schemas."""

    def test_telemetry_schema_defaults(self):
        """Verify default values for TelemetrySchema fields."""
        start_time = time.time()
        schema = TelemetrySchema()
        end_time = time.time()

        self.assertEqual(schema.VERSION, 1)
        self.assertEqual(schema.kind, "TelemetrySchema")
        self.assertGreaterEqual(schema.timestamp, start_time)
        self.assertLessEqual(schema.timestamp, end_time)
        self.assertGreater(len(schema.origin), 0)

    def test_telemetry_schema_custom_fields(self):
        """Verify custom values override schema defaults."""
        schema = TelemetrySchema(
            kind="CustomKind",
            timestamp=12345.678,
            origin="custom_origin",
        )
        self.assertEqual(schema.kind, "CustomKind")
        self.assertEqual(schema.timestamp, 12345.678)
        self.assertEqual(schema.origin, "custom_origin")


class TelemetryEventTest(unittest.TestCase):
    """Unit tests for telemetry events."""

    def test_telemetry_event_defaults(self):
        """Verify TelemetryEvent kind fallback and values mapping."""
        event = TelemetryEvent(values={"key": "value", "count": 42})
        self.assertEqual(event.kind, "TelemetryEvent")
        self.assertEqual(event.values, {"key": "value", "count": 42})

    def test_telemetry_event_custom_fields(self):
        """Verify TelemetryEvent kind when explicitly specified."""
        event = TelemetryEvent(kind="NewGraphAdded", values={"graph_id": "new_object0"})
        self.assertEqual(event.kind, "NewGraphAdded")
        self.assertEqual(event.values, {"graph_id": "new_object0"})

    def test_validate_kind_fallback(self):
        """Verify validate_kind falls back to class name for empty kind."""
        event = TelemetryEvent(kind="")
        self.assertEqual(event.kind, TelemetryEvent.__name__)


class TelemetryInitTest(unittest.TestCase):
    """Unit tests for top-level telemetry module exports and functions."""

    def test_get_telemeter(self):
        """Verify getTelemeter creates a TelemetryPublisher instance."""
        publisher = telemetry.getTelemeter("test_module")
        self.assertIsInstance(publisher, TelemetryPublisher)
        self.assertEqual(publisher.name, "test_module")
        self.assertEqual(publisher.event_logger.name, "telemetry.test_module")


class TelemetryPublisherTest(unittest.TestCase):
    """Unit tests for TelemetryPublisher."""

    def setUp(self):
        self.handler = TelemetryLogHandler()

    def test_publisher_logger_config(self):
        """Verify logger naming, propagation setting, and handlers."""
        publisher = TelemetryPublisher(__name__)
        self.assertEqual(publisher.event_logger.name, f"telemetry.{__name__}")
        self.assertFalse(publisher.event_logger.propagate)
        self.assertTrue(publisher.event_logger.hasHandlers())

    def test_emit_telemetry_events(self):
        """Verify emitting events emits LogRecord with attached telemetry schema."""
        publisher = TelemetryPublisher(__name__)
        publisher.event_logger.setLevel(logging.DEBUG)
        publisher.event_logger.addHandler(self.handler)

        events = [
            (
                publisher.debug,
                logging.DEBUG,
                TelemetryEvent(kind="DebugEvent", values={"step": 1}),
            ),
            (
                publisher.info,
                logging.INFO,
                TelemetryEvent(kind="InfoEvent", values={"step": 2}),
            ),
            (
                publisher.critical,
                logging.CRITICAL,
                TelemetryEvent(kind="CriticalEvent", values={"step": 3}),
            ),
        ]

        for log_func, _, event in events:
            log_func(event)

        self.assertEqual(len(self.handler.records), len(events))

        for record, (_, level, event) in zip(self.handler.records, events):
            self.assertEqual(record.levelno, level)
            self.assertEqual(record.msg, event.kind)
            self.assertEqual(getattr(record, "telemetry_schema", None), event)


class TelemetryIntegrationTest(unittest.TestCase):
    """Integration tests for telemetry emissions during experiment execution."""

    def setUp(self):
        """Set up temporary directory and compose base experiment config."""
        self.output_dir = tempfile.mkdtemp()
        self.handler = TelemetryLogHandler()

        with hydra.initialize_config_dir(version_base=None, config_dir=str(HYDRA_ROOT)):
            self.base_cfg = hydra.compose(
                config_name="experiment",
                overrides=[
                    "experiment=test/profile/base",
                    f"experiment.config.logging.output_dir={self.output_dir}",
                ],
            )

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.output_dir)

    def test_global_matching_step_telemetry(self):
        """Verify GlobalMatchingStep telemetry event is emitted during model step."""
        telemeter = telemetry.getTelemeter(monty_base.__name__)
        telemeter.event_logger.addHandler(self.handler)

        exp = instantiate_experiment(self.base_cfg.experiment)
        with exp:
            exp.experiment_mode = ExperimentMode.TRAIN
            exp.model.set_experiment_mode(exp.experiment_mode)
            exp.pre_epoch()
            exp.pre_episode()
            ctx = RuntimeContext(rng=exp.rng)

            observations, proprioceptive_state = exp.env_interface.step([])
            exp.model.step(ctx, observations, proprioceptive_state)

        records = [r for r in self.handler.records if r.msg == "GlobalMatchingStep"]

        event: TelemetryEvent = records[0].__dict__.get("telemetry_schema")
        self.assertGreater(event.values["monty_matching_steps"], 0)
