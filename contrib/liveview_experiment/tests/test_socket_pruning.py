"""A socket that stops talking must not keep a page — pictures and all — forever."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from contrib.liveview_experiment.src.state_manager import ExperimentStateManager


class FakeSocket:
    """Stands in for pyview's socket: only identity matters to the manager."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"<FakeSocket {self.name}>"


@pytest.fixture
def manager() -> ExperimentStateManager:
    return ExperimentStateManager(route_path="/test")


def register(manager: ExperimentStateManager, name: str, arrived: float) -> FakeSocket:
    socket = FakeSocket(name)
    manager.register_socket(socket)
    manager._socket_arrival[socket] = arrived
    return socket


class TestPruning:
    def test_a_socket_older_than_its_ttl_is_dropped(
        self, manager: ExperimentStateManager
    ) -> None:
        old = register(manager, "old", arrived=0.0)
        fresh = register(manager, "fresh", arrived=manager._socket_ttl_seconds)

        dropped = manager.prune_sockets(now=manager._socket_ttl_seconds + 1)

        assert dropped == 1
        assert manager.connected_sockets == {fresh}
        assert old not in manager._socket_arrival

    def test_a_fresh_socket_is_never_dropped(
        self, manager: ExperimentStateManager
    ) -> None:
        register(manager, "only", arrived=500.0)

        assert manager.prune_sockets(now=501.0) == 0
        assert len(manager.connected_sockets) == 1

    def test_beyond_the_cap_the_oldest_go_first(
        self, manager: ExperimentStateManager
    ) -> None:
        manager._max_sockets = 3

        sockets = [register(manager, f"s{index}", arrived=float(index)) for index in range(6)]

        dropped = manager.prune_sockets(now=10.0)

        assert dropped == 3
        assert manager.connected_sockets == set(sockets[3:])

    def test_pruning_twice_drops_nothing_the_second_time(
        self, manager: ExperimentStateManager
    ) -> None:
        register(manager, "old", arrived=0.0)

        assert manager.prune_sockets(now=10 * manager._socket_ttl_seconds) == 1
        assert manager.prune_sockets(now=10 * manager._socket_ttl_seconds) == 0

    def test_a_socket_that_unregisters_takes_its_arrival_with_it(
        self, manager: ExperimentStateManager
    ) -> None:
        socket = register(manager, "gone", arrived=1.0)

        manager.unregister_socket(socket)

        assert socket not in manager._socket_arrival
