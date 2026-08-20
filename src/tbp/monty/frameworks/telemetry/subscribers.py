# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import abc
import logging
import multiprocessing as mp
import multiprocessing.synchronize
import queue
import threading
from queue import Queue
from typing import final

from tbp.monty.frameworks.telemetry.schemas import TelemetrySchema


class TelemetrySubscriber(logging.Handler, metaclass=abc.ABCMeta):
    """Logging handler that consumes telemetry schemas sent by a `TelemetryPublisher`.

    Schemas are consumed immediately on arrival by the emitting thread. Subclasses must
    implement `logger_names()` and `_consume()`.
    """

    def __init__(self, **kwargs):
        """Base constructor for abstract `TelemetrySubscriber`.

        Args:
            **kwargs: Forwarded to superclass.
        """
        super().__init__(**kwargs)

        self.loggers: set[logging.Logger] = set()
        """Loggers actively subscribed to."""

        self.subscribe()

    def __del__(self):
        self.unsubscribe()

    @property
    @abc.abstractmethod
    def logger_names(self) -> list[str]:
        """Names of loggers to subscribe to."""
        ...

    def subscribe(self):
        """Subscribes the instance to its loggers."""
        for name in self.logger_names:
            logger = logging.getLogger(name)
            logger.addHandler(self)  # idempotent
            self.loggers.add(logger)

    def unsubscribe(self):
        """Unsubscribes the instance from its loggers."""
        for logger in self.loggers:
            logger.removeHandler(self)
        self.loggers.clear()

    @staticmethod
    def _get_schema_from_record(record: logging.LogRecord) -> TelemetrySchema:
        schema = record.__dict__.get("telemetry_schema")
        if not isinstance(schema, TelemetrySchema):
            raise TypeError(f"Telemetry schema of invalid type '{type(schema)}'")
        return schema

    def emit(self, record: logging.LogRecord):
        """Consumes the schema from an incoming telemetry `LogRecord` immediately."""
        schema = self._get_schema_from_record(record)
        self._consume(schema)

    @abc.abstractmethod
    def _consume(self, schema: TelemetrySchema):
        """Processes a single telemetry schema."""
        ...


class ThreadedTelemetrySubscriber(TelemetrySubscriber):
    """Consumes telemetry schemas on a dedicated background thread.

    Incoming schemas are enqueued on arrival and consumed by a daemon thread that blocks
    on each ``get()`` until a ``Queue.shutdown()`` is called.

    Use this for subscribers that don't require main-thread execution (e.g. file
    writers, forwarding bridges).

    For subscribers that must run on the main thread (e.g. matplotlib GUIs), use
    `TelemetrySubscriber` directly.

    Example::

        subscriber = MyThreadedSubscriber()
        subscriber.start()
        # ... schemas consumed in background ...
        subscriber.stop()
    """

    def __init__(self, **kwargs):
        """Initializes the subscriber."""
        super().__init__(**kwargs)

        self.schema_queue: Queue[TelemetrySchema] = queue.Queue()
        """Queue of incoming schemas for background consumption."""

        self._thread = threading.Thread(target=self._consume_loop, daemon=True)
        self._start_flag = threading.Event()

    def __del__(self):
        """Attempts a best-effort stop on garbage collection."""
        self.stop()

    def _clear_queue(self):
        """Clears schema queue."""
        while not self.schema_queue.empty():
            try:
                self.schema_queue.get_nowait()
            except queue.Empty:
                continue
            self.schema_queue.task_done()

    def subscribe(self):
        """Subscribes the instance to its loggers, clearing the schema queue first."""
        self._clear_queue()
        super().subscribe()

    def unsubscribe(self):
        """Unsubscribes the instance from its loggers, clearing the schema queue."""
        super().unsubscribe()
        self._clear_queue()

    def start(self):
        """If not started, subscribes to logger(s) and starts the background thread."""
        if not self._start_flag.is_set():
            self._start_flag.set()
            self.subscribe()
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self):
        """Stops the background thread and unsubscribes from logger(s)."""
        self._start_flag.clear()
        self.unsubscribe()
        if self._thread.is_alive():
            self._thread.join()

    @final
    def emit(self, record: logging.LogRecord):
        """Enqueues the schema from an incoming telemetry `LogRecord`."""
        schema = self._get_schema_from_record(record)
        self.schema_queue.put_nowait(schema)

    def _consume_loop(self):
        """Thread entry point; continuously consumes schemas until stopped."""
        while self._start_flag.is_set():
            try:
                schema = self.schema_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._consume(schema)
            self.schema_queue.task_done()


class MultiprocessTelemetrySubscriber(ThreadedTelemetrySubscriber):
    """Consumes telemetry schemas on a dedicated child process.

    Extends `ThreadedTelemetrySubscriber`, reusing its schema queue, subscription, and
    start/stop mechanics. The thread acts as the bridge between the queue and the child
    process: it pickles incoming schemas and forwards them into an ``mp.Queue`` that the
    child process reads from.

    Subclasses implement `_process_consume()` rather than `_consume()`. The latter is
    reserved for the forwarding logic and must not be overridden.
    """

    def __init__(self, **kwargs):
        """Initializes the subscriber, creating the ``mp.Queue`` and child process.

        The process is not started until `start()` is called.
        """
        super().__init__(**kwargs)
        self._mp_schema_queue: mp.Queue[TelemetrySchema] = mp.Queue()
        self._mp_start_flag = mp.Event()
        self._process = mp.Process(
            target=self._process_main,
            args=(self._mp_schema_queue, self._mp_start_flag),
            daemon=True,
        )

    def _clear_queue(self):
        """Clears schema queues."""
        super()._clear_queue()
        while not self._mp_schema_queue.empty():
            try:
                self._mp_schema_queue.get_nowait()
            except queue.Empty:
                continue

    def start(self):
        """Starts the bridge thread, then the child process."""
        super().start()
        self._mp_start_flag.set()
        if not self._process.is_alive():
            self._process.start()

    def stop(self):
        """Stops the bridge thread, then the child process."""
        self._mp_start_flag.clear()
        super().stop()
        if self._process.is_alive():
            self._process.join()

    @final
    def _consume(self, schema: TelemetrySchema):
        """Pickles and forwards a schema to the child process queue.

        This is the bridge step. Do not override in subclasses; implement
        `_process_consume()` instead.
        """
        self._mp_schema_queue.put(schema)

    def _process_main(self, schema_queue: mp.Queue, start_flag: mp.synchronize.Event):
        """Child process entry point; continuously consumes schemas until stopped."""
        while start_flag.is_set():
            try:
                schema = schema_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._process_consume(schema)

    @abc.abstractmethod
    def _process_consume(self, schema: TelemetrySchema):
        """Schema processing logic. Runs in the child process.

        Note: Cannot reference a live object from the parent process.
        """
        ...
