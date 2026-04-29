# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import unittest
from typing import Any, Callable
from unittest.mock import Mock, sentinel

import numpy as np
import quaternion as qt

from tbp.monty.cmp import Message
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.abstract_monty_classes import SensorObservation
from tbp.monty.frameworks.models.motor_system_state import AgentState, SensorState
from tbp.monty.frameworks.models.two_d_sensor_module import TwoDSensorModule
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.frameworks.utils.edge_detection import DEFAULT_POSE_2D, EdgeFeatures
from tbp.monty.math import DEFAULT_TOLERANCE
from tests.unit.frameworks.utils.edge_detection_test import (
    PATCH_SIZE,
    sensor_observation,
)

DEFAULT_FEATURES = [
    "on_object",
    "pose_vectors",
    "principal_curvatures",
    "edge_strength",
    "coherence",
]
SURFACE_NORMAL_3D = np.array([0.0, 0.0, 1.0])
FLAT_SURFACE_POSE = DEFAULT_POSE_2D


def make_message(
    location: np.ndarray | None = None,
    on_object: bool = True,
    use_state: bool = True,
    pose_vectors: np.ndarray | None = None,
    pose_fully_defined: bool = False,
    principal_curvatures: np.ndarray | None = None,
    non_morphological_features: dict | None = None,
    sender_id: str = "patch",
    sender_type: str = "SM",
):
    if location is None:
        location = np.array([0.0, 0.0, 0.0])
    if pose_vectors is None:
        pose_vectors = np.identity(3)
    if principal_curvatures is None:
        principal_curvatures = np.identity(3)

    morphological_features = {
        "pose_vectors": pose_vectors,
        "pose_fully_defined": pose_fully_defined,
        "on_object": float(on_object),
    }
    non_morphological_features = (
        {} if non_morphological_features is None else non_morphological_features
    )
    if principal_curvatures is not None:
        non_morphological_features["principal_curvatures"] = principal_curvatures

    return Message(
        location=location,
        morphological_features=morphological_features,
        non_morphological_features=non_morphological_features,
        confidence=1.0,
        use_state=use_state,
        sender_id=sender_id,
        sender_type=sender_type,
    )


def make_agent_state(
    sensor_module_id: str = "test",
    agent_position: np.ndarray | None = None,
    agent_rotation: qt.quaternion | None = None,
    sensor_position: np.ndarray | None = None,
    sensor_rotation: qt.quaternion | None = None,
):
    if agent_position is None:
        agent_position = np.zeros(3)
    if agent_rotation is None:
        agent_rotation = qt.quaternion(1, 0, 0, 0)
    if sensor_position is None:
        sensor_position = np.zeros(3)
    if sensor_rotation is None:
        sensor_rotation = qt.quaternion(1, 0, 0, 0)

    return AgentState(
        sensors={
            SensorID(sensor_module_id): SensorState(
                position=sensor_position,
                rotation=sensor_rotation,
            )
        },
        position=agent_position,
        rotation=agent_rotation,
    )


def make_no_edge() -> EdgeFeatures:
    return EdgeFeatures(
        angle=None, strength=0.0, coherence=0.0, is_geometric_edge=False, has_edge=False
    )


def make_raw_observation(
    *, center_location: np.ndarray, semantic_id: int
) -> SensorObservation:
    obs = sensor_observation(angle=None, world_camera=np.identity(4))

    semantic_3d = np.zeros((PATCH_SIZE * PATCH_SIZE, 4), dtype=np.float64)
    semantic_3d[:, :3] = center_location
    semantic_3d[:, 3] = semantic_id

    obs.update(
        semantic_3d=semantic_3d,
        sensor_frame_data=None,
    )
    return obs


def make_2d_sm(
    *,
    sensor_module_id: str = "test",
    features: list[str] | None = None,
    save_raw_obs: bool = False,
    pc1_is_pc2_threshold: int = 10,
    is_surface_sm: bool = False,
    edge_detector: Callable[..., EdgeFeatures] | None = None,
    noise_params: dict[str, Any] | None = None,
    delta_thresholds: dict[str, Any] | None = None,
) -> TwoDSensorModule:
    if features is None:
        features = DEFAULT_FEATURES.copy()

    return TwoDSensorModule(
        sensor_module_id=sensor_module_id,
        features=features,
        save_raw_obs=save_raw_obs,
        pc1_is_pc2_threshold=pc1_is_pc2_threshold,
        is_surface_sm=is_surface_sm,
        edge_detector=edge_detector,
        noise_params=noise_params,
        delta_thresholds=delta_thresholds,
    )


class TwoDSensorModuleInitTest(unittest.TestCase):
    def test_first_observation_initializes_2d_location_from_world_xy(self) -> None:
        two_d_sm = make_2d_sm(edge_detector=Mock(return_value=make_no_edge()))
        world_location = np.array([1.0, 2.0, 3.0])
        percept = make_message(
            location=world_location.copy(), on_object=True, use_state=True
        )
        two_d_sm._observation_processor.process = Mock(return_value=percept)

        msg = two_d_sm.step(
            ctx=RuntimeContext(rng=np.random.RandomState()),
            observation=sentinel.raw_observation,
            motor_only_step=False,
        )

        np.testing.assert_allclose(msg.location, [1.0, 2.0, 0.0])
        np.testing.assert_allclose(two_d_sm._previous_2d_location, [1.0, 2.0])
        np.testing.assert_allclose(two_d_sm._previous_3d_location, world_location)

    def test_like_how_monty_uses_two_d_sm(self) -> None:
        """Mimic how Monty might use SMs in aggregate_sensory_inputs().

        This calls update_state() before step().
        """
        edge_detector = Mock(return_value=make_no_edge())
        two_d_sm = make_2d_sm(
            features=DEFAULT_FEATURES,
            edge_detector=edge_detector,
        )
        agent_state = make_agent_state(sensor_module_id=two_d_sm.sensor_module_id)
        obs = make_raw_observation(
            center_location=np.array([1, 2, 3]),
            semantic_id=1,
        )
        ctx = RuntimeContext(rng=np.random.RandomState())

        two_d_sm.update_state(agent_state)
        msg = two_d_sm.step(ctx, obs, motor_only_step=False)

        assert two_d_sm.state is not None
        assert two_d_sm.states[-1] is two_d_sm.state

        assert msg.sender_id == two_d_sm.sensor_module_id
        assert msg.sender_type == "SM"
        assert msg.confidence == 1.0
        assert isinstance(msg.use_state, bool)

        assert msg.location.shape == (3,)
        assert msg.morphological_features["pose_vectors"].shape == (3, 3)
        assert isinstance(msg.morphological_features["pose_fully_defined"], bool)


class TwoDSensorModuleEdgeTest(unittest.TestCase):
    def test_default_edge_detector_is_used_when_edge_features_requested(self):
        two_d_sm = make_2d_sm(
            sensor_module_id="test",
            features=DEFAULT_FEATURES,
        )

        percept = make_message(
            location=np.array([1.0, 2.0, 3.0]),
            on_object=True,
            use_state=True,
            pose_vectors=np.identity(3),
            pose_fully_defined=False,
            sender_id="test",
        )
        two_d_sm._observation_processor.process = Mock(return_value=percept)

        raw_observation = make_raw_observation(
            center_location=np.array([1.0, 2.0, 3.0]),
            semantic_id=1,
        )

        msg = two_d_sm.step(
            ctx=RuntimeContext(rng=np.random.RandomState()),
            observation=raw_observation,
            motor_only_step=False,
        )

        assert msg.sender_id == "test"
        assert msg.sender_type == "SM"

    def test_edge_detector_not_required_when_edge_features_not_requested(self):
        two_d_sm = TwoDSensorModule(
            sensor_module_id="test",
            features=[
                feature
                for feature in DEFAULT_FEATURES
                if feature not in {"edge_strength", "coherence"}
            ],
        )
        percept = make_message(
            location=np.array([1.0, 2.0, 3.0]),
            on_object=True,
            use_state=True,
            pose_vectors=FLAT_SURFACE_POSE,
            pose_fully_defined=False,
            sender_id="test",
        )
        two_d_sm._observation_processor.process = Mock(return_value=percept)

        msg = two_d_sm.step(
            ctx=RuntimeContext(rng=np.random.RandomState()),
            observation=sentinel.raw_observation,
            motor_only_step=False,
        )

        assert msg.sender_id == "test"
        assert msg.sender_type == "SM"
        assert msg.morphological_features["pose_fully_defined"] is False

    def test_extract_2d_edge_sets_edge_pose_and_features(self):
        observation = make_raw_observation(
            center_location=np.zeros(3),
            semantic_id=1,
        )
        edge_detector = Mock(
            return_value=EdgeFeatures(
                angle=np.pi / 2,
                strength=2.5,
                coherence=0.75,
                is_geometric_edge=False,
                has_edge=True,
            )
        )
        two_d_sm = make_2d_sm(edge_detector=edge_detector)
        two_d_sm._update_tangent_frame(surface_normal_3d=SURFACE_NORMAL_3D)
        percept = make_message()

        msg = two_d_sm._extract_2d_edge(
            percept, observation, surface_normal_3d=SURFACE_NORMAL_3D
        )

        assert msg.morphological_features["pose_fully_defined"] is True

        np.testing.assert_allclose(
            msg.morphological_features["pose_vectors"],
            np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            ),
            atol=DEFAULT_TOLERANCE,
        )

        assert msg.non_morphological_features["edge_strength"] == 2.5
        assert msg.non_morphological_features["coherence"] == 0.75

        edge_detector.assert_called_once_with(observation)

    def test_extract_2d_edge_ignores_no_edge(self):
        observation = make_raw_observation(
            center_location=np.zeros(3),
            semantic_id=1,
        )
        edge_detector = Mock(return_value=make_no_edge())
        two_d_sm = make_2d_sm(edge_detector=edge_detector)
        two_d_sm._update_tangent_frame(surface_normal_3d=SURFACE_NORMAL_3D)
        percept = make_message()
        original_pose = percept.morphological_features["pose_vectors"].copy()

        msg = two_d_sm._extract_2d_edge(percept, observation, SURFACE_NORMAL_3D)

        assert msg.morphological_features["pose_fully_defined"] is False
        np.testing.assert_allclose(
            msg.morphological_features["pose_vectors"], original_pose
        )

    def test_extract_2d_edge_ignores_geometric_edge(self):
        observation = make_raw_observation(
            center_location=np.zeros(3),
            semantic_id=1,
        )
        edge_detector = Mock(
            return_value=EdgeFeatures(
                angle=np.pi / 2,
                strength=2.5,
                coherence=0.75,
                is_geometric_edge=True,
                has_edge=True,
            )
        )
        two_d_sm = make_2d_sm(edge_detector=edge_detector)
        two_d_sm._update_tangent_frame(surface_normal_3d=SURFACE_NORMAL_3D)
        percept = make_message()
        original_pose = percept.morphological_features["pose_vectors"].copy()

        msg = two_d_sm._extract_2d_edge(percept, observation, SURFACE_NORMAL_3D)

        assert msg.morphological_features["pose_fully_defined"] is False
        np.testing.assert_allclose(
            msg.morphological_features["pose_vectors"], original_pose
        )
        assert "edge_strength" not in msg.non_morphological_features
        assert "coherence" not in msg.non_morphological_features


class TwoDSensorModuleTangentFrameTest(unittest.TestCase):
    def test_tangent_frame_transported_not_recreated(self):
        two_d_sm = make_2d_sm(edge_detector=Mock(return_value=make_no_edge()))
        first_normal = np.array([0.0, 0.0, 1.0])
        second_normal = np.array([0.0, np.sqrt(0.5), np.sqrt(0.5)])

        first_pose = np.array(
            [
                first_normal,
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        second_pose = np.array(
            [
                second_normal,
                [1.0, 0.0, 0.0],
                [0.0, np.sqrt(0.5), -np.sqrt(0.5)],
            ]
        )

        two_d_sm._observation_processor.process = Mock(
            side_effect=[
                make_message(
                    location=np.array([0.0, 0.0, 0.0]),
                    pose_vectors=first_pose,
                    on_object=True,
                    use_state=True,
                ),
                make_message(
                    location=np.array([1.0, 0.0, 0.0]),
                    pose_vectors=second_pose,
                    on_object=True,
                    use_state=True,
                ),
            ]
        )

        ctx = RuntimeContext(rng=np.random.RandomState())
        two_d_sm.step(
            ctx=ctx,
            observation=sentinel.raw_observation,
            motor_only_step=False,
        )
        tangent_frame_after_first_step = two_d_sm._tangent_frame

        two_d_sm.step(
            ctx=ctx, observation=sentinel.raw_observation, motor_only_step=False
        )
        tangent_frame_after_second_step = two_d_sm._tangent_frame

        assert tangent_frame_after_second_step is tangent_frame_after_first_step
        np.testing.assert_allclose(
            two_d_sm._tangent_frame.normal,
            second_normal,
            atol=DEFAULT_TOLERANCE,
        )

    def test_multi_step_2d_position_accumulated_on_flat_surface(self):
        two_d_sm = make_2d_sm(edge_detector=Mock(return_value=make_no_edge()))
        ctx = RuntimeContext(rng=np.random.RandomState())

        locations = [
            np.array([1.0, 2.0, 0.0]),
            np.array([2.0, 2.0, 0.0]),
            np.array([2.0, 4.0, 0.0]),
        ]

        messages = [
            make_message(
                location=location.copy(),
                on_object=True,
                use_state=True,
                pose_vectors=FLAT_SURFACE_POSE,
            )
            for location in locations
        ]
        two_d_sm._observation_processor.process = Mock(side_effect=messages)

        outputs = [
            two_d_sm.step(ctx, sentinel.raw_observation, motor_only_step=False)
            for _ in messages
        ]

        np.testing.assert_allclose(two_d_sm._previous_2d_location, [2.0, 4.0])
        np.testing.assert_allclose(two_d_sm._previous_3d_location, [2.0, 4.0, 0.0])

        np.testing.assert_allclose(outputs[0].location, [1.0, 2.0, 0.0])
        np.testing.assert_allclose(outputs[1].location, [2.0, 2.0, 0.0])
        np.testing.assert_allclose(outputs[2].location, [2.0, 4.0, 0.0])
        np.testing.assert_allclose(
            outputs[1].displacement["displacement"], [1.0, 0.0, 0.0]
        )
        np.testing.assert_allclose(
            outputs[2].displacement["displacement"], [0.0, 2.0, 0.0]
        )
