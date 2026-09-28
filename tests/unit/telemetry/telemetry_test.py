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
from typing import cast
from unittest import TestCase

from hypothesis import given
from hypothesis import strategies as st

from tbp.monty import telemetry
from tbp.monty.telemetry.formatters import JsonFormatter
from tbp.monty.telemetry.publishers import TelemetryPublisher
from tbp.monty.telemetry.schemas import TelemetryEvent

simple_string_strategy = st.text(min_size=1, max_size=20)

event_extras_strategy = st.dictionaries(
    keys=simple_string_strategy.filter(lambda k: k not in TelemetryEvent.model_fields),
    values=st.one_of(simple_string_strategy, st.floats(), st.none()),
    max_size=5,
)


class TelemetryLogHandler(logging.Handler):
    """Logging handler that collects log records for telemetry assertions."""

    records: list[logging.LogRecord]

    def __init__(self) -> None:
        super().__init__()
        self.records = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class TelemetryEventTest(TestCase):
    def test_kind_defaults_to_class_name_when_empty(self) -> None:
        event = TelemetryEvent(kind="")
        self.assertEqual(event.kind, event.__class__.__name__)
        self.assertEqual(event.kind, str(event))

    @given(kind=simple_string_strategy)
    def test_kind_is_preserved_when_non_empty(self, kind: str) -> None:
        event = TelemetryEvent(kind=kind)
        self.assertEqual(event.kind, kind)
        self.assertEqual(event.kind, str(event))

    @given(kind=simple_string_strategy, event_extras=event_extras_strategy)
    def test_custom_fields_override_kind(self, kind: str, event_extras: dict) -> None:
        event = TelemetryEvent(kind=kind, **event_extras)
        self.assertEqual(event.kind, kind)
        self.assertEqual(event.model_extra, event_extras)


class JsonFormatterTest(TestCase):
    record_strategy = st.builds(
        logging.LogRecord,
        name=simple_string_strategy,
        level=st.integers(),
        pathname=st.just(__file__),
        lineno=st.integers(),
        msg=st.builds(
            lambda kind, event_extras: TelemetryEvent(kind=kind, **event_extras),
            simple_string_strategy,
            event_extras_strategy,
        ),
        args=st.just(()),
        exc_info=st.none(),
        func=simple_string_strategy,
    )

    def setUp(self) -> None:
        self.formatter = JsonFormatter()

    @given(record=record_strategy)
    def test_format_logrecord_to_expected_json(self, record: logging.LogRecord) -> None:
        data = json.loads(self.formatter.format(record))
        expected = {
            "level": logging.getLevelName(record.levelno),
            "name": record.name,
            "funcName": record.funcName,
            "lineno": record.lineno,
            **cast("TelemetryEvent", record.msg).model_dump(mode="json"),
        }
        self.assertEqual(data, expected)

    @given(record=record_strategy)
    def test_raise_typeerror_if_msg_is_not_telemetry_schema(
        self, record: logging.LogRecord
    ) -> None:
        record.msg = "I am not a schema"
        with self.assertRaises(TypeError):
            self.formatter.format(record)


class GetTelemeterTest(TestCase):
    def test_returns_publisher_with_telemetry_prefixed_name(self) -> None:
        telemeter = telemetry.getTelemeter(__name__)
        self.assertIsInstance(telemeter, TelemetryPublisher)
        self.assertEqual(telemeter.name, f"telemetry.{__name__}")
        self.assertFalse(telemeter.propagate)

    def test_does_not_duplicate_telemetry_prefix(self) -> None:
        telemeter = telemetry.getTelemeter(f"telemetry.{__name__}")
        self.assertIsInstance(telemeter, TelemetryPublisher)
        self.assertEqual(telemeter.name, f"telemetry.{__name__}")
        self.assertFalse(telemeter.propagate)


class TelemetryPublisherTest(TestCase):
    def setUp(self) -> None:
        self.telemeter = telemetry.getTelemeter(__name__)
        self.telemeter.setLevel(logging.NOTSET)
        self.handler = TelemetryLogHandler()

    def tearDown(self) -> None:
        self.telemeter.removeHandler(self.handler)

    def test_logrecord_msg_is_telemetry_event(self) -> None:
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
