from typing import Any, Dict, TypedDict

import glom
import numpy as np

from tbp.monty.frameworks.models.motor_system import MotorSystemState


class MS(Dict[str, Any]):
    def get_agent_state(self, agent_id: str) -> Any:
        return self[agent_id]

    def set_agent_state(self, agent_id: str, state: Any) -> None:
        self[agent_id] = state


ms = MS()
ms["agent_id_0"] = {
    "rotation": np.array([1, 0, 0, 0]),
    "position": np.array([0, 0, 0]),
    "sensors": {
        "sensor_id_0": {
            "rotation": np.array([1, 0, 0, 0]),
            "position": np.array([0, 0, 0]),
        }
    },
}
print(ms.get_agent_state("agent_id_0"))
agent_rot = glom.glom(ms, "agent_id_0.rotation")
sensor_pos = glom.glom(ms, "agent_id_0.sensors.sensor_id_0.position")
print(agent_rot)
print(sensor_pos)

m = MotorSystemState()
