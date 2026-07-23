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
import queue
import threading
from queue import Queue
from typing import final

from tbp.monty.frameworks.telemetry.schemas import TelemetrySchema, TelemetryStopEvent


class TelemetryConsumer(logging.Handler, metaclass=abc.ABCMeta):
    """Logging handler that consumes telemetry schemas sent by a `TelemetryEmitter`.

    If ``deferred=False``, schemas are consumed immediately on arrival by the emitting
    thread.

    If ``deferred=True``, incoming schemas are enqueued for later consumption. The
    caller drives consumption by calling `pump()` periodically, typically from the main
    thread interleaved with other work (see example 2).

    Subclasses must implement `logger_names()` and `_consume()`. Override `_post_pump()`
    to run custom logic after consumption, e.g. ``plt.pause()``.

    Example 1::

        consumer = MyConsumer()  # non-deferred = self-pumping

    Example 2::

        consumer = MyConsumer(deferred=True)
        while not done:
            ...  # (your work goes here)
            consumer.pump()
    """

    def __init__(self, deferred=False, **kwargs):
        """Base constructor for abstract `TelemetryConsumer`.

        Args:
            deferred: If ``False``, schemas are consumed immediately on arrival by the
                emitting thread. If ``True``, incoming schemas are enqueued for later
                consumption via `pump()`.
            **kwargs: Forwarded to superclass.
        """
        super().__init__(**kwargs)
        self.deferred = deferred

        self.loggers: set[logging.Logger] = set()
        """Loggers actively subscribed to."""

        self.schema_queue: Queue[TelemetrySchema] = queue.Queue()
        """Queue of incoming schemas for deferred consumption."""

        self.subscribe()

    def __del__(self):
        self.unsubscribe()

    @property
    @abc.abstractmethod
    def logger_names(self) -> list[str]:
        """Names of loggers to subscribe to."""
        ...

    def subscribe(self):
        """Subscribes the consumer to its loggers."""
        self._clear_queue()
        for name in self.logger_names:
            logger = logging.getLogger(name)
            logger.addHandler(self)  # idempotent
            self.loggers.add(logger)

    def unsubscribe(self, consume=False):
        """Unsubscribes the consumer from its loggers.

        Args:
            consume: If ``True``, consumes remaining queued schemas.
        """
        for logger in self.loggers:
            logger.removeHandler(self)
        self.loggers.clear()

        if consume:
            self.pump()
        else:
            self._clear_queue()

    def _clear_queue(self):
        """Clears schema queue."""
        with self.schema_queue.mutex:
            self.schema_queue.queue.clear()

    def emit(self, record: logging.LogRecord):
        """Consumes or enqueues the schema from an incoming telemetry `LogRecord`.

        If ``self.deferred`` is ``False``, schemas are consumed immediately on arrival.
        Otherwise, they are enqueued for later consumption via `pump()`.

        Raises:
            TypeError: If invalid schema type received.
        """
        schema: TelemetrySchema = record.__dict__.get("telemetry_schema")
        if not isinstance(schema, TelemetrySchema):
            raise TypeError(f"Telemetry schema of invalid type '{type(schema)}'")

        if self.deferred:
            self.schema_queue.put_nowait(schema)
        else:
            self._consume(schema)

    def pump(self, continuous=False, wait=False):
        """Consumes pending schemas from deferred queue, then calls `_post_pump()`.

        No effect if ``self.deferred`` is ``False``. Otherwise, in non-continuous mode
        (default), consumes all queued schemas and returns. In continuous mode, blocks
        on each `get()` until a `TelemetryStopEvent` is received; used internally by
        `ThreadedTelemetryConsumer._pump_loop()`.

        `_post_pump()` is called once after the queue is consumed. Not called if a
        `TelemetryStopEvent` causes an early return.

        Args:
            continuous: If ``True``, keeps waiting for new schemas once the queue is
                empty, instead of returning. Raises error if on main thread.
            wait: If ``continuous=False`` and queue is empty, waits for the next schema,
                consumes it, then returns. Waits indefinitely in the absence of any
                incoming schema. No effect if ``continuous=True``.

        Raises:
            RuntimeError: If called with ``continuous=True`` on main thread.
        """
        if continuous and threading.current_thread() is threading.main_thread():
            raise RuntimeError("Consumer cannot pump continuously on main thread")

        while True:
            if continuous:
                schema = self.schema_queue.get()
            else:
                try:
                    schema = self.schema_queue.get_nowait()
                except queue.Empty:
                    if wait:
                        schema = self.schema_queue.get()
                        wait = False
                    else:
                        break
            if isinstance(schema, TelemetryStopEvent):
                return
            self._consume(schema)
        self._post_pump()

    def flush(self):
        self.pump()

    @abc.abstractmethod
    def _consume(self, schema: TelemetrySchema):
        """Processes a single telemetry schema."""
        ...

    def _post_pump(self):
        """Called once after `pump()` consumes the queue in non-continuous mode.

        Override to add custom post-consumption logic. Not called when `pump()` exits
        early due to a `TelemetryStopEvent`.

        Example::

            def _post_pump(self):
                plt.pause(0.00001)
        """
        return


class ThreadedTelemetryConsumer(TelemetryConsumer, abc.ABC):
    """A `TelemetryConsumer` that drives `pump()` on a dedicated background thread.

    Starts a daemon thread that calls ``pump(continuous=True)``, blocking on each schema
    until a `TelemetryStopEvent` is received. The calling thread is never blocked by
    schema processing.

    Use this for consumers that don't require main-thread execution (e.g. file writers,
    forwarding bridges).

    For consumers that must run on the main thread (e.g. matplotlib GUIs), use
    `TelemetryConsumer` directly.

    Example::

        consumer = MyThreadedConsumer()
        consumer.start()
        # ... schemas consumed in background ...
        consumer.stop()
    """

    def __init__(self, **kwargs):
        """Initializes the consumer."""
        super().__init__(deferred=True, **kwargs)
        self._thread = threading.Thread(target=self._pump_loop, daemon=True)

    def __del__(self):
        """Attempts a best-effort stop on garbage collection."""
        self.stop()

    def start(self):
        """Subscribes to the logger(s) and start the background thread.

        No effect if already running.
        """
        if not self._thread.is_alive():
            self.subscribe()
            self._thread.start()

    def stop(self):
        """Signals the background thread to stop and join it.

        Sends a `TelemetryStopEvent` to unblock the thread if it is waiting on
        ``get()``, then joins it. Unsubscribes from the logger(s) unconditionally.
        """
        if self._thread.is_alive():
            self.schema_queue.put(TelemetryStopEvent(emitter=self.__class__.__name__))
            self._thread.join()
        self.unsubscribe()

    def _pump_loop(self):
        """Thread entry point. Continuously pumps the queue until stopped."""
        self.pump(continuous=True)


class MultiprocessTelemetryConsumer(ThreadedTelemetryConsumer, abc.ABC):
    """Telemetry consumer that runs a dedicated child process.

    Extends `ThreadedTelemetryConsumer`, reusing its schema queue, bridge thread,
    subscribe/unsubscribe, and stop sentinel machinery. The inherited thread acts as the
    bridge between the queue and the child process: it consumes the queued schemas and
    forwards them into an ``mp.Queue`` that the child process reads from.

    Subclasses implement `_process_consume()` rather than `_consume()`. The latter is
    reserved for the forwarding logic and must not be overridden.
    """

    def __init__(self, **kwargs):
        """Initializes the consumer, creating the ``mp.Queue`` and child process.

        The process is not started until `start()` is called.
        """
        super().__init__(**kwargs)
        self._mp_queue: mp.Queue = mp.Queue()
        self._process = mp.Process(
            target=self._process_main, args=(self._mp_queue,), daemon=True
        )

    def start(self):
        """Starts the child process, then the bridge thread.

        Process is started first so it is ready to receive schemas as soon as the bridge
        thread begins forwarding.
        """
        self._process.start()
        super().start()  # starts bridge thread + subscribe

    def stop(self):
        """Stops the bridge thread, then the child process.

        Joins the bridge thread first via ``super()`` to ensure all pending schemas have
        been forwarded to the queue before the sentinel is sent to the child process.
        """
        super().stop()  # consumes queue, joins bridge thread, unsubscribes
        if self._process.is_alive():
            self._mp_queue.put(TelemetryStopEvent(emitter=self.__class__.__name__))
            self._process.join()

    @final
    def _consume(self, schema: TelemetrySchema):
        """Forwards a schema to the child process queue.

        This is the bridge step. Do not override in subclasses; implement
        `_process_consume()` instead.
        """
        self._mp_queue.put(schema)

    def _process_main(self, mp_queue: mp.Queue):
        """Entry point for the child process.

        Consumes the queue in a blocking loop, calling `_process_consume()` for each
        schema until `TelemetryStopEvent` is received.
        """
        while True:
            schema = mp_queue.get()
            if isinstance(schema, TelemetryStopEvent):
                break
            self._process_consume(schema)

    @abc.abstractmethod
    def _process_consume(self, schema: TelemetrySchema):
        """Schema processing logic. Runs in the child process.

        Note: Cannot reference a live object from the parent process.
        """
        ...
