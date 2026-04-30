import numpy as np
import pytest

from tbp.monty.frameworks.models.evidence_matching.feature_evidence.calculator import (
    DefaultFeatureEvidenceCalculator,
)
from tbp.monty.frameworks.models.sensor_modules import CameraSM
from tbp.monty.frameworks.models.abstract_monty_classes import SensorObservation
from tbp.monty.frameworks.models.motor_system_state import SensorState
from tbp.monty.context import RuntimeContext
from tbp.monty.cmp import Message


def test_lbp_feature_evidence_with_mock_values():
    """
    Test LBP feature evidence calculation with mock values.
    
    This test sets up a mock sensor observation with a specific 
    RGBA input and an expected LBP histogram.
    It then runs the CameraSM to extract the LBP features and uses the 
    DefaultFeatureEvidenceCalculator to compute evidence values.

    Note: This test assumes that the local binary pattern is being computed
    using "uniform" method, R = 1 and P = 8.
    """
    # --- Setup semantic obs ---
    obs3d = np.zeros((4096, 4), dtype=np.float64)
    obs3d[2080] = [0.0, 0.0, 0.0, 1.0]

    # --- Sensor observation ---
    sensor_observation = SensorObservation(
        sensor_module_id="camera_lbp_enabled",
        rgba=np.array([
            [[123, 45, 67, 255], [12, 200, 34, 128], [255, 0, 89, 255], [34, 67, 210, 90], [88, 12, 45, 255]],
            [[90, 123, 56, 200], [11, 22, 33, 44], [222, 111, 0, 255], [76, 54, 32, 128], [9, 8, 7, 255]],
            [[255, 255, 0, 255], [0, 128, 255, 180], [45, 67, 89, 255], [210, 45, 67, 220], [134, 156, 178, 255]],
            [[12, 34, 56, 78], [98, 76, 54, 255], [33, 66, 99, 150], [255, 100, 50, 255], [0, 0, 0, 0]],
            [[77, 88, 99, 255], [123, 234, 45, 200], [67, 89, 123, 255], [210, 210, 210, 255], [1, 2, 3, 255]]
        ], dtype=np.uint8),
        semantic_3d=obs3d,
        depth=np.zeros((64, 64), dtype=np.float64),
        sensor_frame_data=np.zeros((4096, 4), dtype=np.float64),
        world_camera=np.zeros((4, 4), dtype=np.float64),
    )

    # --- Expected LBP histogram ---
    expected_hist = np.array([
        0.24, 0.08, 0.16, 0.08, 0.00,
        0.04, 0.12, 0.04, 0.12, 0.12
    ], dtype=np.float32)

    # --- Node feature vectors (normalized) ---
    arrs = np.array([
        [0.0227, 0.1645, 0.0851, 0.0624, 0.1720, 0.1059, 0.1475, 0.0416, 0.1209, 0.0737],
        [0.1448, 0.0297, 0.1825, 0.0952, 0.1329, 0.0417, 0.1667, 0.0714, 0.1171, 0.0198],
        [0.0100, 0.1317, 0.0578, 0.1617, 0.0938, 0.1856, 0.0280, 0.0758, 0.1437, 0.1198],
        [0.1774, 0.0484, 0.1028, 0.0181, 0.1532, 0.0826, 0.1269, 0.0342, 0.1916, 0.0605],
        [0.0685, 0.1591, 0.0222, 0.1168, 0.0524, 0.1815, 0.0887, 0.1371, 0.0040, 0.1677],
    ], dtype=np.float32)

    # --- Run sensor module ---
    sm = CameraSM(
        sensor_module_id="camera_lbp_enabled",
        features=["local_binary_pattern"],
        save_raw_obs=True,
    )
    sm.is_exploring = False
    sm.state = SensorState(position=np.zeros(3), rotation=np.zeros(3))

    context = RuntimeContext(rng=np.random.RandomState(42))
    msg: Message = sm.step(context, sensor_observation)

    lbp = msg.non_morphological_features["local_binary_pattern"]

    # --- Assert LBP extraction ---
    np.testing.assert_allclose(lbp, expected_hist, atol=1e-5)

    # --- Evidence: near-zero tolerance → all zero ---
    evidence_zero = DefaultFeatureEvidenceCalculator.calculate(
        channel_feature_array=arrs,
        channel_feature_order=["local_binary_pattern"],
        channel_feature_weights={"local_binary_pattern": 1.0},
        channel_query_features={"local_binary_pattern": lbp},
        channel_tolerances={"local_binary_pattern": 1e-6},
        input_channel="camera_lbp_enabled",
    )

    assert np.allclose(evidence_zero, 0)

    # --- Evidence: moderate tolerance ---
    evidence_half = DefaultFeatureEvidenceCalculator.calculate(
        channel_feature_array=arrs,
        channel_feature_order=["local_binary_pattern"],
        channel_feature_weights={"local_binary_pattern": 1.0},
        channel_query_features={"local_binary_pattern": lbp},
        channel_tolerances={"local_binary_pattern": 0.5},
        input_channel="camera_lbp_enabled",
    )

    expected_half = np.array([
        0.55029199, 0.72934136, 0.44204575, 0.726612, 0.45024061
    ])

    np.testing.assert_allclose(evidence_half, expected_half, atol=1e-5)