# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import json
import logging
from unittest import TestCase

import hydra

from tbp.monty import telemetry
from tbp.monty.frameworks.experiments.monty_experiment import MontyExperiment
from tbp.monty.hydra import instantiate_experiment
from tbp.monty.telemetry.formatters import JsonFormatter
from tbp.monty.telemetry.publishers import TelemetryPublisher
from tbp.monty.telemetry.schemas import TelemetryEvent
from tests import HYDRA_ROOT


def clear_all_loggers():
    logging.Logger.manager.loggerDict.clear()


class TelemetryLogHandler(logging.Handler):
    """Logging handler that collects log records for telemetry assertions."""

    records: list[logging.LogRecord]

    def __init__(self) -> None:
        super().__init__()
        self.records = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class TelemetryEventTest(TestCase):
    def test_kind_defaults_to_class_name(self) -> None:
        event = TelemetryEvent(kind="")
        self.assertEqual(event.kind, event.__class__.__name__)
        self.assertEqual(event.kind, str(event))

    def test_custom_fields_override_kind(self) -> None:
        kind = "NewGraphAdded"
        graph_id = "new_object0"
        event = TelemetryEvent(kind=kind, graph_id=graph_id)
        self.assertEqual(event.kind, kind)
        self.assertEqual(event.graph_id, graph_id)


class JsonFormatterTest(TestCase):
    def test_format_logrecord_to_expected_json(self) -> None:
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


class GetTelemeterTest(TestCase):
    def test_returns_publisher_with_telemetry_prefixed_name(self) -> None:
        telemeter = telemetry.getTelemeter(f"telemetry.tbp.monty.{__name__}")
        self.assertIsInstance(telemeter, TelemetryPublisher)
        self.assertEqual(telemeter.name, f"telemetry.tbp.monty.{__name__}")
        self.assertFalse(telemeter.propagate)


class TelemetryPublisherTest(TestCase):
    def setUp(self) -> None:
        clear_all_loggers()
        self.handler = TelemetryLogHandler()
        self.telemeter = telemetry.getTelemeter(f"tbp.monty.{__name__}")
        self.telemeter.setLevel(logging.NOTSET)

    def tearDown(self) -> None:
        self.telemeter.removeHandler(self.handler)

    def test_info_raises_typeerror_if_not_event_type(self) -> None:
        with self.assertRaises(TypeError):
            self.telemeter.info("")

    def test_logs_log_records_with_telemetry_event_as_the_message(self) -> None:
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
            self.assertEqual(record.levelno, level)
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

    def test_hydra_info_config(self) -> None:
        with self._instantiate_experiment("info"):
            self.telemeter.addHandler(self.handler)
            self.telemeter.debug(TelemetryEvent(kind="TestEvent"))
            self.telemeter.info(TelemetryEvent(kind="TestEvent"))

            # Validate loglevel compliance
            self.assertEqual(self.telemeter.getEffectiveLevel(), logging.INFO)
            self.assertEqual(len(self.handler.records), 1)

            # Validate presence of telemetry_console handler and telemetry_formatter
            logger = logging.getLogger("telemetry.tbp.monty")
            handler = next(
                (h for h in logger.handlers if isinstance(h, logging.StreamHandler)),
                None,
            )
            self.assertIsNotNone(handler)
            self.assertIsInstance(handler.formatter, JsonFormatter)
