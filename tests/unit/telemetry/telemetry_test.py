# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import io
import json
import logging
import unittest

import hydra
import pytest

from tbp.monty import telemetry
from tbp.monty.frameworks.experiments.monty_experiment import MontyExperiment
from tbp.monty.frameworks.models import monty_base
from tbp.monty.hydra import instantiate_experiment
from tbp.monty.telemetry.formatters import JsonFormatter
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


class LoggerTestCase(unittest.TestCase):
    def setUp(self):
        # Clean up all loggers
        logging.Logger.manager.loggerDict.clear()


class TelemetryEventTest(LoggerTestCase):
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


class JsonFormatterTest(LoggerTestCase):
    """Unit tests for `JsonFormatter`."""

    def setUp(self):
        super().setUp()
        telemetry.getTelemeter("tbp.monty").setLevel(logging.NOTSET)
        self.telemeter = telemetry.getTelemeter(monty_base.__name__)
        self.telemeter.setLevel(logging.NOTSET)

    def test_format(self):
        """Verify `JsonFormatter` formats a LogRecord into expected JSON."""
        event = TelemetryEvent(kind="CustomEvent", graph_id="obj_1")
        record = logging.LogRecord(
            name="telemetry.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=42,
            msg=event,
            args=(),
            exc_info=None,
            func="test_format",
        )

        formatter = JsonFormatter()
        output = formatter.format(record)
        data = json.loads(output)

        expected = {
            "level": "INFO",
            "name": "telemetry.test",
            "funcName": "test_format",
            "lineno": 42,
            **event.model_dump(mode="json"),
        }
        self.assertEqual(data, expected)

    def test_json_formatter(self):
        """Verify `JsonFormatter` formats output when used by a logging handler."""
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonFormatter())

        self.telemeter.setLevel(logging.INFO)
        self.telemeter.addHandler(handler)
        try:
            event = TelemetryEvent(kind="StreamEvent", graph_id="obj_1")
            self.telemeter.info(event)
            output = stream.getvalue()
            data = json.loads(output)

            self.assertEqual(data["level"], "INFO")
            self.assertEqual(data["kind"], "StreamEvent")
            self.assertEqual(data["graph_id"], "obj_1")
        finally:
            self.telemeter.removeHandler(handler)


class TelemetryPublisherTest(LoggerTestCase):
    """Unit tests for `TelemetryPublisher`."""

    def setUp(self):
        super().setUp()
        telemetry.getTelemeter("tbp.monty").setLevel(logging.NOTSET)
        self.handler = TelemetryLogHandler()
        self.telemeter = telemetry.getTelemeter(monty_base.__name__)
        self.telemeter.setLevel(logging.NOTSET)

    def tearDown(self):
        self.telemeter.removeHandler(self.handler)

    def test_params(self):
        """Verify logger params."""
        self.assertIsInstance(self.telemeter, TelemetryPublisher)
        self.assertEqual(self.telemeter.name, f"telemetry.{monty_base.__name__}")
        self.assertFalse(self.telemeter.propagate)

    def test_events(self):
        """Verify emitting events produces log records with telemetry schemas."""
        self.telemeter.setLevel(logging.DEBUG)
        self.telemeter.addHandler(self.handler)

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
            log_func(event, stack_info=True)

        self.assertEqual(len(self.handler.records), len(events))

        for record, (_, level, event) in zip(self.handler.records, events):
            print(record.stack_info)
            self.assertEqual(record.levelno, level)
            self.assertEqual(str(record.msg), event.kind)
            self.assertIs(record.msg, event)

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
            self.telemeter.addHandler(self.handler)
            self.telemeter.debug(TelemetryEvent(kind="TestEvent"))
            self.assertEqual(self.telemeter.getEffectiveLevel(), logging.DEBUG)
            self.assertEqual(len(self.handler.records), 1)

    def test_info_config(self):
        """Verify behavior of Hydra config ``telemetry=info``."""
        with self._instantiate_experiment("info"):
            self.telemeter.addHandler(self.handler)
            self.telemeter.debug(TelemetryEvent(kind="TestEvent"))
            self.telemeter.info(TelemetryEvent(kind="TestEvent"))
            self.assertEqual(self.telemeter.getEffectiveLevel(), logging.INFO)
            self.assertEqual(len(self.handler.records), 1)

    def test_warning_config(self):
        """Verify behavior of Hydra config ``telemetry=warning``."""
        with self._instantiate_experiment("warning"):
            self.telemeter.addHandler(self.handler)
            self.telemeter.warning(TelemetryEvent(kind="TestEvent"))
            self.assertEqual(len(self.handler.records), 1)

    def test_error_config(self):
        """Verify behavior of Hydra config ``telemetry=error``."""
        with self._instantiate_experiment("error"):
            self.telemeter.addHandler(self.handler)
            self.telemeter.error(TelemetryEvent(kind="TestEvent"))
            self.assertEqual(len(self.handler.records), 1)

    def test_critical_config(self):
        """Verify behavior of Hydra config ``telemetry=critical``."""
        with self._instantiate_experiment("critical"):
            self.telemeter.addHandler(self.handler)
            self.telemeter.critical(TelemetryEvent(kind="TestEvent"))
            self.assertEqual(len(self.handler.records), 1)
