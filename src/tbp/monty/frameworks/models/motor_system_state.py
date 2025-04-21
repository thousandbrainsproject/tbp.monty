# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from typing import Any, Dict, TypedDict

import numpy as np


class SensorState(TypedDict):
    """The proprioceptive state of a sensor.

    TODO: Change into dataclass
    """

    position: Any  # TODO: Stop using magnum.Vector3 and decide on Monty standard
    """The sensor's position relative to the agent."""
    rotation: Any  # TODO: Stop using quaternion.quaternion and decide on Monty standard
    """The sensor's rotation relative to the agent."""


class AgentState(TypedDict):
    """The proprioceptive state of an agent.

    TODO: Change into dataclass
    """

    sensors: Dict[str, SensorState]
    """The proprioceptive state of the agent's sensors."""
    position: Any  # TODO: Stop using magnum.Vector3 and decide on Monty standard
    """The agent's position relative to some global reference frame."""
    rotation: Any  # TODO: Stop using quaternion.quaternion and decide on Monty standard
    """The agent's rotation relative to some global reference frame."""


class ProprioceptiveState(Dict[str, AgentState]):
    """The proprioceptive state of the motor system.

    TODO: Change into dataclass
    """


class MotorSystemState(Dict[str, Any]):
    """The state of the motor system.

    TODO: Currently, ProprioceptiveState can be cast to MotorSystemState since
          MotorSystemState is a generic dictionary. In the future, make
          ProprioceptiveState a param on MotorSystemState to more clearly distinguish
          between the two.
    """

    def convert_motor_state(self):
        """Convert the motor state into something that can be pickled/saved to JSON.

        i.e. substitute vector and quaternion objects; note e.g. copy.deepcopy does not
        work.

        TODO ?clean this up with a recursive algorithm, or use BufferEncoder in
        buffer.py

        Returns:
            (dict): Copy of the motor state.
        """
        state_copy = {}
        for key in self.keys():
            state_copy[key] = {}
            for key_inner in self[key].keys():
                if type(self[key][key_inner]) is dict:
                    state_copy[key][key_inner] = {}
                    # We need to go deeper
                    for key_inner_inner in self[key][key_inner].keys():
                        state_copy[key][key_inner][key_inner_inner] = {}
                        if type(self[key][key_inner][key_inner_inner]) is dict:
                            # We need to go even deeper...
                            # (**Hans Zimmer music intensifies**)
                            for key_i_i_i in self[key][key_inner][key_inner_inner]:
                                state_copy[key][key_inner][key_inner_inner][
                                    key_i_i_i
                                ] = {}
                                try:
                                    state_copy[key][key_inner][key_inner_inner][
                                        key_i_i_i
                                    ] = np.array(
                                        list(
                                            self[key][key_inner][key_inner_inner][
                                                key_i_i_i
                                            ]
                                        )
                                    )
                                except TypeError:
                                    # Quaternions
                                    state_copy[key][key_inner][key_inner_inner][
                                        key_i_i_i
                                    ] = [
                                        self[key][key_inner][key_inner_inner][
                                            key_i_i_i
                                        ].real
                                    ] + list(
                                        self[key][key_inner][key_inner_inner][
                                            key_i_i_i
                                        ].imag
                                    )
                elif type(self[key][key_inner]) is bool:
                    pass
                else:
                    try:
                        state_copy[key][key_inner] = np.array(
                            list(self[key][key_inner])
                        )
                    except TypeError:
                        # Quaternions
                        state_copy[key][key_inner] = [self[key][key_inner].real] + list(
                            self[key][key_inner].imag
                        )
        return state_copy
