from __future__ import annotations

import unittest

from tbp.monty.frameworks.environment_utils import transforms
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.frameworks.models.abstract_monty_classes import (
    AgentObservations,
    Observations,
    SensorObservations,
)
import numpy as np

from dataclasses import dataclass
from tbp.monty.frameworks.models.motor_system_state import ProprioceptiveState

AGENT_ID = AgentID("camera")
SENSOR_ID = SensorID("sensor_01")

@dataclass
class TransformContext:
    rng: np.random.RandomState
    state: ProprioceptiveState | None = None

TEST_OBS = Observations(
    {
        AGENT_ID: AgentObservations(
            {
                SENSOR_ID: SensorObservations(
                    {
                        "semantic": np.array(
                            [
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 5, 5, 5, 5, 0, 0],
                                [0, 0, 5, 5, 5, 5, 0, 0],
                                [0, 0, 5, 5, 5, 5, 0, 0],
                                [0, 0, 5, 5, 5, 5, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                            ],
                            dtype=int,
                        ),
                        "depth": np.array(
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0],
                                [0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0],
                                [0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0],
                                [0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            ]
                        ),
                    }
                )
            }
        )
    }
)

class TransformPipelineInstantiationTest(unittest.TestCase):
    def test_empty_pipeline(self):
        transform = transforms.TransformPipeline([])
        transform(TEST_OBS, TransformContext(rng=np.random.RandomState(42)))

class SingleTransformTest(unittest.TestCase):
    def test_missing_max_depth(self):
        missing_to_max_depth_args = {"agent_id": AGENT_ID, "max_depth": 10.0}
        transform = transforms.TransformPipeline([transforms.TransformMiddleware(transforms.MissingToMaxDepth, **missing_to_max_depth_args)])
        transform(TEST_OBS, TransformContext(rng=np.random.RandomState(42)))

class TransformChainTest(unittest.TestCase):
    def test_missing_max_depth_to_add_noise(self):
        missing_to_max_depth_args = {"agent_id": AGENT_ID, "max_depth": 10.0}
        add_noise_to_raw_depth_image_args = {"agent_id": AGENT_ID, "sigma": 1.0}
        transform = transforms.TransformPipeline(
            [transforms.TransformMiddleware(transforms.MissingToMaxDepth, **missing_to_max_depth_args),
             transforms.TransformMiddleware(transforms.AddNoiseToRawDepthImage, **add_noise_to_raw_depth_image_args)])
        transform(TEST_OBS, TransformContext(rng=np.random.RandomState(42)))

class DepthTo3DLocationsTest(unittest.TestCase):
    def test_missing_max_depth_to_depth_to_3d(self):
        resolution = TEST_OBS[AGENT_ID][SENSOR_ID]["depth"].shape
        missing_to_max_depth_args = {"agent_id": AGENT_ID, "max_depth": 10.0}
        depth_to_3d_args = {
            "agent_id": AGENT_ID,
            "sensor_ids": [SENSOR_ID],
            "resolutions": [resolution],
            "use_semantic_sensor": True
        }
        transform = transforms.TransformPipeline([transforms.TransformMiddleware(transforms.MissingToMaxDepth, **missing_to_max_depth_args),
                                                  transforms.TransformMiddleware(transforms.DepthTo3DLocations, **depth_to_3d_args)])
        transform(TEST_OBS, TransformContext(rng=np.random.RandomState(42)))

if __name__ == "__main__":
    unittest.main()