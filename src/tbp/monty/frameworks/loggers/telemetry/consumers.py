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
import multiprocessing as mp
import queue
import threading
from queue import Queue
from typing import ClassVar, final

from tbp.monty.frameworks.loggers.telemetry.events import (
    TELEMETRY_STOP,
    TelemetryEvent,
    TelemetryStopEvent,
)
from tbp.monty.frameworks.loggers.telemetry.producers import TelemetryBroker


class TelemetryConsumer(abc.ABC):
    """Base class for telemetry consumers driven by an external `pump()` call.

    Maintains a `queue.Queue` subscribed to the broker for the declared `schema_ids`.
    The caller drives consumption by calling pump() periodically, typically from the
    main thread interleaved with other work or a GUI loop.

    Subclasses declare `schema_ids` and implement `_consume()`. Override `_post_pump()`
    for logic to run after each drain, e.g. `plt.pause()`.

    Usage::
        consumer = MyConsumer(broker)
        consumer.subscribe()
        while not done:
            consumer.pump()
        consumer.pump()  # final drain
        consumer.unsubscribe()
    """

    schema_ids: ClassVar[list[str]]  # subclass declares which schemas it wants

    def __init__(self, broker: TelemetryBroker, **kwargs):
        """Base constructor for abstract `TelemetryConsumer`.

        Args:
            broker: The `TelemetryBroker` to subscribe to.
            **kwargs: Forwarded to superclass.
        """
        super().__init__(**kwargs)
        self._broker = broker
        self.event_queue: Queue[TelemetryEvent] = queue.Queue()

    def subscribe(self):
        """Registers `event_queue` with the broker for `schema_ids`.

        Clears any stale events from a previous run before re-registering.
        """
        with self.event_queue.mutex:
            self.event_queue.queue.clear()
        self._broker.subscribe(schema_ids=self.schema_ids, event_queue=self.event_queue)

    def unsubscribe(self):
        """Deregisters `event_queue` from the broker for `schema_ids`."""
        self._broker.unsubscribe(self.schema_ids, self.event_queue)

    def pump(self, continuous=False):
        """Consumes pending events from the queue, then calls `_post_pump()`.

        In non-continuous mode (default), drains all queued events and returns.
        In continuous mode, blocks on each get() until a `TelemetryStopEvent` is
        received; used internally by `ThreadedTelemetryConsumer._pump_loop()`.

        `_post_pump()` is called once after the queue drains. Not called if a
        TelemetryStopEvent causes an early return.
        """
        while True:
            if continuous:
                event = self.event_queue.get()
            else:
                try:
                    event = self.event_queue.get_nowait()
                except queue.Empty:
                    break
            if isinstance(event, TelemetryStopEvent):
                return
            self._consume(event)
        self._post_pump()

    @abc.abstractmethod
    def _consume(self, event: TelemetryEvent):
        """Processes a single telemetry event."""
        ...

    def _post_pump(self):
        """Called once after `pump()` drains the queue in non-continuous mode.

        Not called when `pump()` exits early due to a `TelemetryStopEvent`.
        Override to add post-drain behavior,

        Example::

            def _post_pump(self):
                plt.pause(0.00001)
        """
        return


class ThreadedTelemetryConsumer(TelemetryConsumer, abc.ABC):
    """A `TelemetryConsumer` that drives `pump()` on a dedicated background thread.

    Starts a daemon thread that calls `pump(continuous=True)`, blocking on each event
    until a `TelemetryStopEvent` is received. The calling thread is never blocked by
    event processing.

    Use this for consumers whose `_consume()` is thread-safe and does not require
    main-thread execution (e.g. file writers, forwarding bridges).

    For consumers that must run on the main thread (e.g. matplotlib GUIs), use
    `TelemetryConsumer` directly instead and call `pump()` from the main thread.

    Usage::

        consumer = MyThreadedConsumer(broker)
        consumer.start()
        # ... events processed in background ...
        consumer.stop()
    """

    def __init__(self, broker: TelemetryBroker, **kwargs):
        """Initializes the consumer.

        Args:
            broker: The TelemetryBroker to subscribe to.
            **kwargs: Forwarded to TelemetryConsumer.
        """
        super().__init__(broker=broker, **kwargs)
        self._thread = threading.Thread(target=self._pump_loop, daemon=True)

    def __del__(self):
        """Attempts a best-effort stop on garbage collection."""
        self.stop()

    def start(self):
        """Subscribes to the broker and start the background thread.

        No effect if already running.
        """
        if not self._thread.is_alive():
            self.subscribe()
            self._thread.start()

    def stop(self):
        """Signals the background thread to stop and join it.

        Sends a TelemetryStopEvent to unblock the thread if it is waiting on `get()`,
        then joins it. Unsubscribes from the broker unconditionally.
        """
        if self._thread.is_alive():
            self.event_queue.put(TELEMETRY_STOP)
            self._thread.join()
        self.unsubscribe()

    def _pump_loop(self):
        """Thread entry point. Runs pump(continuous=True) until stopped."""
        self.pump(continuous=True)


class MultiprocessTelemetryConsumer(ThreadedTelemetryConsumer, abc.ABC):
    """Telemetry consumer that consumes events in a dedicated child process.

    Extends `ThreadedTelemetryConsumer`, reusing its broker queue, bridge thread,
    subscribe/unsubscribe, and stop sentinel machinery. The inherited thread acts as the
    bridge between the broker queue and the child process: it drains the broker's
    `queue.Queue` and forwards events into a `multiprocessing.Queue` that the child
    process reads from.

    Subclasses implement `_process_consume()` rather than `_consume()`. The latter is
    reserved for the forwarding logic and must not be overridden.
    """

    def __init__(self, broker: TelemetryBroker, **kwargs):
        """Initializes the consumer, creating the `mp.Queue` and child process.

        The process is not started until `start()` is called.

        Args:
            broker: The `TelemetryBroker` to subscribe to.
            **kwargs: Forwarded to superclass.
        """
        super().__init__(broker=broker, **kwargs)
        self._mp_queue: mp.Queue = mp.Queue()
        self._process = mp.Process(
            target=self._process_main, args=(self._mp_queue,), daemon=True
        )

    def start(self):
        """Starts the child process, then the bridge thread.

        Process is started first so it is ready to receive events as soon as the bridge
        thread begins forwarding.
        """
        self._process.start()
        super().start()  # starts bridge thread + subscribe

    def stop(self):
        """Stops the bridge thread, then the child process.

        Joins the bridge thread first via `super()` to ensure all pending events have
        been forwarded to _mp_queue before the sentinel is sent to the child process.
        """
        super().stop()  # drains broker queue, joins bridge thread, unsubscribes
        if self._process.is_alive():
            self._mp_queue.put(TELEMETRY_STOP)
            self._process.join()

    @final
    def _consume(self, event: TelemetryEvent):
        """Forwards an event from the broker queue to the child process.

        This is the bridge step. Do not override in subclasses; implement
        `_process_consume()` instead.
        """
        self._mp_queue.put(event)

    def _process_main(self, mp_queue: mp.Queue):
        """Entry point for the child process.

        Drains `mp_queue` in a blocking loop, calling `_process_consume()` for each
        event until a TelemetryStopEvent is received.
        """
        while True:
            event = mp_queue.get()
            if isinstance(event, TelemetryStopEvent):
                break
            self._process_consume(event)

    @abc.abstractmethod
    def _process_consume(self, event: TelemetryEvent):
        """Event processing logic. Runs in the child process.

        Note: Cannot reference a live object from the parent process.
        """
        ...
