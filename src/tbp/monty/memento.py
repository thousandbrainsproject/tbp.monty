# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Any, Mapping, Protocol

Memento = Mapping[str, Any]


class Snapshotable(Protocol):
    """Objects that can save/load their internal state.

    The internal state of the object is represented by a memento,
    which can be treated as an opaque token used to reset the object
    to the state it had at the time of the snapshot.
    """

    def state_dict(self) -> Memento:
        """Snapshot a memento representing the internal state of this object.

        Returns:
            State dict for logging and saving.
        """
        ...

    def load_state_dict(self, memento: Memento) -> None:
        """Load the internal object state from a previously snapshot memento.

        Args:
            memento: State dict to load.
        """
        ...
