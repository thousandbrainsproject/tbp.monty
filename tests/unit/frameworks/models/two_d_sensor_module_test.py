# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from typing import Any, Callable
from unittest.mock import Mock, sentinel

import numpy as np
import pytest
from hypothesis import given
from vtkmodules import qt

from tbp.monty.cmp import Message
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.abstract_monty_classes import SensorObservation
from tbp.monty.frameworks.models.motor_system_state import AgentState, SensorState
from tbp.monty.frameworks.models.two_d_sensor_module import TwoDSensorModule
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.frameworks.utils.edge_detection import EdgeFeatures
from tbp.monty.math import DEFAULT_TOLERANCE
from tests.unit.frameworks.utils.edge_detection_test import (
    PATCH_SIZE,
    sensor_observation,
)

DEFAULT_FEATURES = [
    "pose_vectors",
    "principal_curvatures",
    "edge_strength",
    "coherence",
]


def make_message(
    location: np.ndarray = np.array([0.0, 0.0, 0.0]),
    on_object: bool = True,
    use_state: bool = True,
    pose_vectors: np.ndarray = np.identity(3),
    pose_fully_defined: bool = False,
    principal_curvatures: np.ndarray = np.identity(3),
    non_morphological_features: dict | None = None,
    sender_id: str = "patch",
    sender_type: str = "SM",
):
    morphological_features = {
        "pose_vectors": pose_vectors,
        "pose_fully_defined": pose_fully_defined,
        "on_object": float(on_object),
    }
    non_morphological_features = (
        {} if non_morphological_features is None else non_morphological_features
    )
    if principal_curvatures is None:
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

    if edge_detector is None:
        edge_detector = Mock(return_value=make_no_edge())

    two_d_sm = TwoDSensorModule(
        sensor_module_id=sensor_module_id,
        features=features,
        save_raw_obs=save_raw_obs,
        pc1_is_pc2_threshold=pc1_is_pc2_threshold,
        is_surface_sm=is_surface_sm,
        edge_detector=edge_detector,
        noise_params=noise_params,
        delta_thresholds=delta_thresholds,
    )
    return two_d_sm


"""Ideas on what to test:
1. `test_step_snapshots_raw_observation_as_needed`
    - Idea taken from SalienceSMTest.
    - Implemented for practice (and I think I have something a bit better).
    - But I think this applies to all SensorModule and may not belong in this file.
    - If so, delete implementation and save it to Shortcut Ticket.

2. Percept Filter/`delta_thresholds`
    - Idea was to assert that first percept passes, unchanged percepts become use_state=False,
    significant distance/feature change passes, and pre_episode() resets the filter. 
    - Not implemented as it is generic to both CameraSM and TwoDSensorModule.
    - Again, likely a candidate for Shortcut ticket.
    
3. Noise with RuntimeContext.rng
    - Not implemented as it is generic to both CameraSM and TwoDSensorModule.
    - Honestly have not decided on how noise (in locations, especially) may affect
    accumulated 2D trajectory. 
    
4. `test_pre_episode_resets_all_state`
    - Idea: Run N on object steps to pretend that we are at the end of an episode
    - Call pre_episode()
    - Assert states are reset

5. `test_update_state_computes_sensor_world_position`
    - Make an AgentState with known rotation and sensor offset.
    - Call update_state()
    - Make sure sm.state.position matches equation.
    - Make sure sm.state.rotation matches equation.

6. `test_first_observation_initialies_2d_location_from_world_xy`
    - Make an on object step at random 3d location [x, y, z].
    - Make sure msg.location == [x, y, 0] (z forced to zero)
    - Make sure _previous_2d_location == [x, y]
    - Make sure _previous_3d_location == [x, y, z].

7. `test_mutli_step_2d_position_accumulated_on_flat_surface`
    - Pretend to take 3 steps on flat surface (surface_normal = [0, 0, 1])
    - Assert that after 3 steps, _previous_2d_location has accumulated appropriate displacement

8. `test_tangent_frame_transported_not_recreated`
    - Pretend to take two on-object steps with slightly different surface normals.
    - Assert _tangent_frame identity _is_ the same before and after.
    - Assert _tangent_frame.normal matches the surface normal after the steps.

9. `test_off_object_step`
    - Pretend that first step is on_object, and second step is off object
    - The displacement after Step 2 should be [0, 0, 0]
    - Assert _previous_2d_location is unchanged from after Step 1.
    - Assert _previous_3d_location is unchanged from after Step 1.

10. `test_motor_only_step_sets_use_state_false`
    - Take one on-object step with motor_only_step=True.
    - Assert:
        Returned msg.use_state is False.
        Observation IS still appended to processed_obs (storage is independent of use_state).

11. `test_is_exploring`
    Set is_exploring=True, run a step.

    Assert:
        processed_obs == [] and states == [].
        Sub-test with is_exploring=False: len(processed_obs) == 1.

12. `test_state_dict_contains_processed_observations`
    - Run two steps, call state_dict()
    - Assert that result has "processed_observations" key with two entries.
"""


@pytest.mark.parametrize(
    ("save_raw_obs", "is_exploring", "expected_snapshots"),
    [
        (True, False, 1),
        (True, True, 0),
        (False, False, 0),
        (False, True, 0),
    ],
)
def test_step_snapshots_raw_observation_as_needed(
    save_raw_obs: bool,
    is_exploring: bool,
    expected_snapshots: bool,
):
    # I updated slightly from SalienceSMTest because
    # I think this mirrors how the codebase might use a SM
    # (not by checking whether a method was called). 🤷‍♀️
    two_d_sm = make_2d_sm(save_raw_obs=save_raw_obs)
    two_d_sm.is_exploring = is_exploring
    two_d_sm.update_state(make_agent_state())

    ctx = RuntimeContext(rng=np.random.RandomState())
    obs = sentinel.raw_observation
    two_d_sm._observation_processor = Mock(return_value=make_message())

    two_d_sm.step(ctx, obs)
    state = two_d_sm.state_dict()
    assert len(state["raw_observation"]) == expected_snapshots
    assert len(state["sm_properties"]) == expected_snapshots

    if expected_snapshots:
        assert state["raw_observation"][0] is obs
        np.testing.assert_allclose(
            state["sm_properties"][0]["sm_location"],
            two_d_sm.state.position,
        )
        np.testing.assert_allclose(
            state["sm_properties"][0]["sm_rotation"],
            qt.as_float_array(two_d_sm.state.rotation),
        )


def test_pre_episode_resets_all_state(save_raw):
    two_d_sm = make_2d_sm()
    assert two_d_sm._previous_3d_location is None
    np.testing.assert_allclose(
        two_d_sm._previous_2d_location, [0.0, 0.0], atol=DEFAULT_TOLERANCE
    )
    assert two_d_sm.tangent_frame is None
    assert two_d_sm.processed_obs == []
    assert two_d_sm.states == []
    assert two_d_sm.is_exploring is False


@given(obs=sensor_observation(patterns=["horizontal_edge"]))
def test_basic_step(obs):
    obs.update(
        semantic_3d=np.ones((PATCH_SIZE * PATCH_SIZE, 4), dtype=int),
        sensor_frame_data=None,
    )
    sm = TwoDSensorModule("test", features=DEFAULT_FEATURES)
    msg = sm.step(ctx=Mock(), observation=obs, motor_only_step=False)
    print(msg)


SURFACE_NORMAL_3D = np.array([0.0, 0.0, 1.0])


def test_extract_2d_edge_sets_edge_pose_and_features():
    observation = SensorObservation(
        world_camera=sentinel.world_camera,
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
    state = make_message()

    result = two_d_sm._extract_2d_edge(
        state, observation, surface_normal_3d=SURFACE_NORMAL_3D
    )

    assert result is state
    assert result.morphological_features["pose_fully_defined"] is True

    np.testing.assert_allclose(
        result.morphological_features["pose_vectors"],
        np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        atol=DEFAULT_TOLERANCE,
    )

    assert result.non_morphological_features["edge_strength"] == 2.5
    assert result.non_morphological_features["coherence"] == 0.75

    edge_detector.assert_called_once_with(
        observation,
        surface_normal=SURFACE_NORMAL_3D,
    )


def test_extract_2d_edge_ignores_no_edge():
    observation = SensorObservation(
        world_camera=sentinel.world_camera,
    )
    edge_detector = Mock(return_value=make_no_edge())
    two_d_sm = make_2d_sm(edge_detector=edge_detector)
    two_d_sm._update_tangent_frame(surface_normal_3d=SURFACE_NORMAL_3D)
    state = make_message()
    original_pose = state.morphological_features["pose_vectors"].copy()

    result = two_d_sm._extract_2d_edge(state, observation, SURFACE_NORMAL_3D)

    assert result.morphological_features["pose_fully_defined"] is False
    np.testing.assert_allclose(
        result.morphological_features["pose_vectors"], original_pose
    )


def test_extract_2d_edge_ignores_geometric_edge():
    observation = SensorObservation(
        world_camera=sentinel.world_camera,
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
    state = make_message()
    original_pose = state.morphological_features["pose_vectors"].copy()

    result = two_d_sm._extract_2d_edge(state, observation, SURFACE_NORMAL_3D)

    assert result.morphological_features["pose_fully_defined"] is False
    np.testing.assert_allclose(
        result.morphological_features["pose_vectors"], original_pose
    )
    assert "edge_strength" not in result.non_morphological_features
    assert "coherence" not in result.non_morphological_features
