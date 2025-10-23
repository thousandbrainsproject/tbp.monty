# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os
from dataclasses import dataclass, field
from numbers import Number
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.environment_utils.transforms import (
    DepthTo3DLocations,
)
from tbp.monty.frameworks.environments.two_d_data import (
    OmniglotEnvironment,
    SaccadeOnImageEnvironment,
    SaccadeOnImageFromStreamEnvironment,
)
from tbp.monty.frameworks.environments.ycb import SHUFFLED_YCB_OBJECTS
from tbp.monty.frameworks.utils.transform_utils import scipy_to_numpy_quat

# ---------
# run / training / eval args
# ---------


@dataclass
class ExperimentArgs:
    do_train: bool = True
    do_eval: bool = True
    show_sensor_output: bool = False
    max_train_steps: int = 1000
    max_eval_steps: int = 500
    max_total_steps: int = 4 * (max_train_steps + max_eval_steps)  # Total number of
    # episode steps that can be taken before timing out, regardless of e.g. whether LMs
    # receive sensory information and therefore perform a true matching step (due to
    # e.g. being off the object)
    n_train_epochs: int = 3
    n_eval_epochs: int = 3
    model_name_or_path: str = ""
    min_lms_match: int = 1
    seed: int = 42


@dataclass
class DebugExperimentArgs(ExperimentArgs):
    do_train: bool = True
    do_eval: bool = True
    max_train_steps: int = 50
    max_eval_steps: int = 50
    max_total_steps: int = 4 * (max_train_steps + max_eval_steps)
    n_train_epochs: int = 1
    n_eval_epochs: int = 1


@dataclass
class EvalExperimentArgs(ExperimentArgs):
    do_train: bool = False
    n_eval_epochs: int = 1
    python_log_level: str = "DEBUG"


# Data-set containing RGBD images of real-world objects taken with a mobile device
@dataclass
class EnvInitArgsMontyWorldStandardScenes:
    data_path: str = os.path.join(
        os.environ["MONTY_DATA"], "worldimages/standard_scenes/"
    )


@dataclass
class EnvInitArgsMontyWorldBrightScenes:
    data_path: str = os.path.join(
        os.environ["MONTY_DATA"], "worldimages/bright_scenes/"
    )


@dataclass
class EnvInitArgsMontyWorldDarkScenes:
    data_path: str = os.path.join(os.environ["MONTY_DATA"], "worldimages/dark_scenes/")


# Data-set where a hand is prominently visible holding (and thereby partially
# occluding) the objects
@dataclass
class EnvInitArgsMontyWorldHandIntrusionScenes:
    data_path: str = os.path.join(
        os.environ["MONTY_DATA"], "worldimages/hand_intrusion_scenes/"
    )


# Data-set where there are two objects in the image; the target class is in the centre
# of the image; of the 4 images per target, the first two images contain similar objects
# (e.g. another type of mug), while the last two images contain distinct objects (such
# as a book if the target is a type of mug)
@dataclass
class EnvInitArgsMontyWorldMultiObjectScenes:
    data_path: str = os.path.join(
        os.environ["MONTY_DATA"], "worldimages/multi_object_scenes/"
    )


@dataclass
class OmniglotDatasetArgs:
    env_init_func: Callable = field(default=OmniglotEnvironment)
    env_init_args: Dict = field(default_factory=dict)
    transform: Union[Callable, list, None] = None

    def __post_init__(self):
        self.transform = [
            DepthTo3DLocations(
                agent_id="agent_id_0",
                sensor_ids=["patch"],
                resolutions=np.array([[10, 10]]),
                world_coord=True,
                zooms=1,
                get_all_points=True,
                use_semantic_sensor=False,
                depth_clip_sensors=(0,),
                clip_value=1.1,
            ),
        ]


@dataclass
class WorldImageDatasetArgs:
    env_init_func: Callable = field(default=SaccadeOnImageEnvironment)
    env_init_args: Dict = field(
        default_factory=lambda: EnvInitArgsMontyWorldStandardScenes().__dict__
    )
    transform: Union[Callable, list, None] = None


@dataclass
class WorldImageFromStreamDatasetArgs:
    env_init_func: Callable = field(default=SaccadeOnImageFromStreamEnvironment)
    env_init_args: Dict = field(default_factory=dict)
    transform: Union[Callable, list, None] = None

    def __post_init__(self):
        self.transform = [
            DepthTo3DLocations(
                agent_id="agent_id_0",
                sensor_ids=["patch"],
                resolutions=np.array([[64, 64]]),
                world_coord=True,
                zooms=1,
                get_all_points=True,
                hfov=11,
                use_semantic_sensor=False,
                depth_clip_sensors=(0,),
                clip_value=1.1,
            ),
            # GaussianSmoothing(agent_id="agent_id_0", sigma=8, kernel_width=10),
        ]


@dataclass
class DefaultTrainObjectList:
    objects: List[str] = field(default_factory=lambda: SHUFFLED_YCB_OBJECTS[0:2])


@dataclass
class DefaultEvalObjectList:
    objects: List[str] = field(default_factory=lambda: SHUFFLED_YCB_OBJECTS[2:6])


@dataclass
class NotYCBTrainObjectList:
    objects: List[str] = field(
        default_factory=lambda: [
            "capsule3DSolid",
            "coneSolid",
            "cubeSolid",
        ]
    )


@dataclass
class NotYCBEvalObjectList:
    objects: List[str] = field(
        default_factory=lambda: [
            "coneSolid",
            "cubeSolid",
            "cylinderSolid",
        ]
    )


class DefaultObjectInitializer:
    def __call__(self):
        euler_rotation = self.rng.uniform(0, 360, 3)
        q = Rotation.from_euler("xyz", euler_rotation, degrees=True).as_quat()
        quat_rotation = scipy_to_numpy_quat(q)
        return dict(
            rotation=quat_rotation,
            euler_rotation=euler_rotation,
            position=(self.rng.uniform(-0.5, 0.5), 0.0, 0.0),
            scale=[1.0, 1.0, 1.0],
        )

    def post_epoch(self):
        pass

    def post_episode(self):
        pass

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class PredefinedObjectInitializer(DefaultObjectInitializer):
    def __init__(
        self, positions=None, rotations=None, scales=None, change_every_episode=None
    ):
        # NOTE: added param change_every_episode. This is so if I want to run an
        # experiment and specify an exact list of objects, with specific poses per
        # object, I can set this to True. Otherwise I have to loop over all objects
        # for every pose specified.
        self.positions = positions or [[0.0, 1.5, 0.0]]
        self.rotations = rotations or [[0.0, 0.0, 0.0], [45.0, 0.0, 0.0]]
        self.scales = scales or [[1.0, 1.0, 1.0]]
        self.current_epoch = 0
        self.current_episode = 0
        self.change_every_episode = change_every_episode

    def __call__(self):
        mod_counter = (
            self.current_episode if self.change_every_episode else self.current_epoch
        )
        q = Rotation.from_euler(
            "xyz",
            self.rotations[mod_counter % len(self.rotations)],
            degrees=True,
        ).as_quat()
        quat_rotation = scipy_to_numpy_quat(q)
        return dict(
            rotation=quat_rotation,
            euler_rotation=self.rotations[mod_counter % len(self.rotations)],
            quat_rotation=q,
            position=self.positions[mod_counter % len(self.positions)],
            scale=self.scales[mod_counter % len(self.scales)],
        )

    def __repr__(self):
        string = "PredefinedObjectInitializer with params: \n"
        string += f"\t positions: {self.positions}\n"
        string += f"\t rotations: {self.rotations}\n"
        string += f"\t change every episode: {self.change_every_episode}"
        return string

    def __len__(self):
        return len(self.all_combinations_of_params())

    def post_epoch(self):
        self.current_epoch += 1

    def post_episode(self):
        self.current_episode += 1

    def all_combinations_of_params(self):
        param_list = []
        for i in range(len(self.rotations)):
            for j in range(len(self.scales)):
                for k in range(len(self.positions)):
                    params = dict(
                        rotations=[self.rotations[i]],
                        scales=[self.scales[j]],
                        positions=[self.positions[k]],
                    )
                    param_list.append(params)
        return param_list


class RandomRotationObjectInitializer(DefaultObjectInitializer):
    def __init__(self, position=None, scale=None):
        if position is not None:
            self.position = position
        else:
            self.position = [0.0, 1.5, 0.0]
        if scale is not None:
            self.scale = scale
        else:
            self.scale = [1.0, 1.0, 1.0]

    def __call__(self):
        euler_rotation = self.rng.uniform(0, 360, 3)
        q = Rotation.from_euler("xyz", euler_rotation, degrees=True).as_quat()
        quat_rotation = scipy_to_numpy_quat(q)
        return dict(
            rotation=quat_rotation,
            euler_rotation=euler_rotation,
            quat_rotation=q,
            position=self.position,
            scale=self.scale,
        )


@dataclass
class EnvironmentDataloaderPerObjectArgs:
    object_names: List
    object_init_sampler: Callable


@dataclass
class EnvironmentDataLoaderPerObjectTrainArgs(EnvironmentDataloaderPerObjectArgs):
    object_names: List = field(default_factory=lambda: DefaultTrainObjectList().objects)
    object_init_sampler: Callable = field(default_factory=DefaultObjectInitializer)


@dataclass
class EnvironmentDataLoaderPerObjectEvalArgs(EnvironmentDataloaderPerObjectArgs):
    object_names: List = field(default_factory=lambda: DefaultTrainObjectList().objects)
    object_init_sampler: Callable = field(default_factory=DefaultObjectInitializer)


@dataclass
class FixedRotationEnvironmentDataLoaderPerObjectTrainArgs(
    EnvironmentDataloaderPerObjectArgs
):
    object_names: List = field(default_factory=lambda: DefaultTrainObjectList().objects)
    object_init_sampler: Callable = field(
        default_factory=lambda: PredefinedObjectInitializer()
    )


@dataclass
class FixedRotationEnvironmentDataLoaderPerObjectEvalArgs(
    EnvironmentDataloaderPerObjectArgs
):
    object_names: List = field(default_factory=lambda: DefaultTrainObjectList().objects)
    object_init_sampler: Callable = field(
        default_factory=lambda: PredefinedObjectInitializer()
    )


@dataclass
class EnvironmentDataloaderMultiObjectArgs:
    object_names: Dict  # Note Dict and not List
    object_init_sampler: Callable


def get_object_names_by_idx(
    start, stop, list_of_indices=None, object_list=SHUFFLED_YCB_OBJECTS
):
    if isinstance(list_of_indices, list):
        if len(list_of_indices) > 0:
            return [object_list[i] for i in list_of_indices]

    else:
        return object_list[start:stop]


def get_env_dataloader_per_object_by_idx(start, stop, list_of_indices=None):
    return EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(start, stop, list_of_indices),
        object_init_sampler=PredefinedObjectInitializer(),
    )


@dataclass
class OmniglotDataloaderArgs:
    """Set basic debug args to load 3 characters of 2 alphabets in 1 version."""

    alphabets: List = field(default_factory=lambda: [0, 0, 0, 1, 1, 1])
    characters: List = field(default_factory=lambda: [1, 2, 3, 1, 2, 3])
    versions: List = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])


@dataclass
class WorldImageDataloaderArgs:
    """Set basic debug args to load 1 scene (Numenta mug) in 4 versions."""

    scenes: List = field(default_factory=lambda: [0, 0, 0, 0])
    versions: List = field(default_factory=lambda: [0, 1, 2, 3])


def get_omniglot_train_dataloader(num_versions, alphabet_ids, data_path=None):
    """Generate OmniglotDataloaderArgs automatically for training.

    Args:
        num_versions: Number of versions to show for each character (starting at 1).
        alphabet_ids: IDs of alphabets to show. All characters within an
            alphabet will be presented which may be a variable amount.
        data_path: path to the omniglot dataset. If none its set to
            ~/tbp/data/omniglot/python/

    Returns:
        OmniglotDataloaderArgs for training.
    """
    if data_path is None:
        data_path = os.path.join(os.environ["MONTY_DATA"], "omniglot/python/")
    if os.path.exists(data_path):
        alphabet_folders = [
            a for a in os.listdir(data_path + "images_background") if a[0] != "."
        ]
    else:
        # Use placeholder here to pass Circle CI config check.
        return OmniglotDataloaderArgs()
    all_alphabet_idx = []
    all_character_idx = []
    all_version_idx = []
    for a_idx in alphabet_ids:
        alphabet = alphabet_folders[a_idx]
        characters_in_a = list(os.listdir(data_path + "images_background/" + alphabet))
        for c_idx, character in enumerate(characters_in_a):
            versions_of_char = list(
                os.listdir(
                    data_path + "images_background/" + alphabet + "/" + character
                )
            )
            for v_idx in range(len(versions_of_char)):
                if v_idx < num_versions:
                    all_alphabet_idx.append(a_idx)
                    all_character_idx.append(c_idx + 1)
                    all_version_idx.append(v_idx + 1)

    return OmniglotDataloaderArgs(
        alphabets=all_alphabet_idx,
        characters=all_character_idx,
        versions=all_version_idx,
    )


def get_omniglot_eval_dataloader(
    start_at_version, alphabet_ids, num_versions=None, data_path=None
):
    """Generate OmniglotDataloaderArgs automatically for evaluation.

    Args:
        start_at_version: Version number of character to start at. Then shows all
            the remaining versions.
        alphabet_ids: IDs of alphabets to test. Tests all characters within
            the alphabet.
        num_versions: Number of versions of each character to test. If None, all
            versions are shown.
        data_path: path to the omniglot dataset. If none its set to
            ~/tbp/data/omniglot/python/

    Returns:
        OmniglotDataloaderArgs for evaluation.
    """
    if data_path is None:
        data_path = os.path.join(os.environ["MONTY_DATA"], "omniglot/python/")
    if os.path.exists(data_path):
        alphabet_folders = [
            a for a in os.listdir(data_path + "images_background") if a[0] != "."
        ]
    else:
        # Use placeholder here to pass Circle CI config check.
        return OmniglotDataloaderArgs()
    all_alphabet_idx = []
    all_character_idx = []
    all_version_idx = []
    for a_idx in alphabet_ids:
        alphabet = alphabet_folders[a_idx]
        characters_in_a = list(os.listdir(data_path + "images_background/" + alphabet))
        for c_idx, character in enumerate(characters_in_a):
            if num_versions is None:
                versions_of_char = list(
                    os.listdir(
                        data_path + "images_background/" + alphabet + "/" + character
                    )
                )
                num_versions = len(versions_of_char) - start_at_version

            for v_idx in range(num_versions + start_at_version):
                if v_idx >= start_at_version:
                    all_alphabet_idx.append(a_idx)
                    all_character_idx.append(c_idx + 1)
                    all_version_idx.append(v_idx + 1)

    return OmniglotDataloaderArgs(
        alphabets=all_alphabet_idx,
        characters=all_character_idx,
        versions=all_version_idx,
    )


@dataclass
class SensorAgentMapping:
    agent_ids: List[str]
    sensor_ids: List[str]
    sensor_to_agent: Dict


@dataclass
class SingleSensorAgentMapping(SensorAgentMapping):
    """Mapping for a sim with a single agent and single sensor."""

    agent_ids: List[str] = field(default_factory=lambda: ["agent_id_0"])
    sensor_ids: List[str] = field(default_factory=lambda: ["sensor_id_0"])
    sensor_to_agent: Dict = field(
        default_factory=lambda: dict(sensor_id_0="agent_id_0")
    )


@dataclass
class SimpleMountSensorAgentMapping(SensorAgentMapping):
    """Mapping for a sim with a single mount agent with two sensors."""

    agent_ids: List[str] = field(default_factory=lambda: ["agent_id_0"])
    sensor_ids: List[str] = field(
        default_factory=lambda: ["sensor_id_0", "sensor_id_1"]
    )
    sensor_to_agent: Dict = field(
        default_factory=lambda: dict(sensor_id_0="agent_id_0", sensor_id_1="agent_id_0")
    )


@dataclass
class PatchAndViewSensorAgentMapping(SensorAgentMapping):
    agent_ids: List[str] = field(default_factory=lambda: ["agent_id_0"])
    sensor_ids: List[str] = field(default_factory=lambda: ["patch", "view_finder"])
    sensor_to_agent: Dict = field(
        default_factory=lambda: dict(patch="agent_id_0", view_finder="agent_id_0")
    )


@dataclass
class TwoCameraMountConfig:
    agent_id: Union[str, None] = field(default=None)
    sensor_ids: Union[List[str], None] = field(default=None)
    resolutions: List[List[Union[int, float]]] = field(
        default_factory=lambda: [[16, 16], [16, 16]]
    )
    positions: List[List[Union[int, float]]] = field(
        default_factory=lambda: [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]]
    )
    rotations: List[List[Union[int, float]]] = field(
        default_factory=lambda: [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    )
    semantics: List[List[Union[int, float]]] = field(
        default_factory=lambda: [False, False]
    )
    zooms: List[float] = field(default_factory=lambda: [1.0, 1.0])

    def __post_init__(self):
        if self.agent_id is None and self.sensor_ids is None:
            sensor_agent_mapping = SimpleMountSensorAgentMapping()
            self.agent_id = sensor_agent_mapping.agent_ids[0]
            self.sensor_ids = sensor_agent_mapping.sensor_ids


@dataclass
class PatchAndViewFinderMountConfig:
    """Config using view finder to find the object before starting the experiment.

    A common default for Viviane's experiments that use the view finder to navigate
    so the object is in view before the real experiment happens.
    """

    agent_id: Union[str, None] = "agent_id_0"
    sensor_ids: Union[List[str], None] = field(
        default_factory=lambda: ["patch", "view_finder"]
    )
    height: Union[float, None] = 0.0
    position: List[Union[int, float]] = field(default_factory=lambda: [0.0, 1.5, 0.2])
    resolutions: List[List[Union[int, float]]] = field(
        default_factory=lambda: [[64, 64], [64, 64]]
    )
    positions: List[List[Union[int, float]]] = field(
        default_factory=lambda: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    rotations: List[List[Union[int, float]]] = field(
        default_factory=lambda: [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    )
    semantics: List[List[Union[int, float]]] = field(
        default_factory=lambda: [False, False]
    )
    zooms: List[float] = field(default_factory=lambda: [10.0, 1.0])


@dataclass
class PatchAndViewFinderMountLowResConfig(PatchAndViewFinderMountConfig):
    resolutions: List[List[Union[int, float]]] = field(
        default_factory=lambda: [[5, 5], [64, 64]]
    )


@dataclass
class SurfaceAndViewFinderMountConfig(PatchAndViewFinderMountConfig):
    """Use surface agent and view finder to find the object before experiment start.

    Adaptation of Viviane's code that use the view finder to navigate so
    the object is in view before the real experiment happens + surface-agent sensor
    """

    # The height should be zero, so that body actions do not cause the agent
    # to move backward or forward
    height: Union[float, None] = 0.0
    position: List[Union[int, float]] = field(default_factory=lambda: [0.0, 1.5, 0.1])
    resolutions: List[List[Union[int, float]]] = field(
        default_factory=lambda: [[64, 64], [64, 64]]
    )
    # The surface sensor is at the same position as the agent, the viewfinder is 3cm
    # behind the agent. So when the agent turns, the viewfinder moves accordingly.
    positions: List[List[Union[int, float]]] = field(
        default_factory=lambda: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.03]]
    )
    zooms: List[float] = field(default_factory=lambda: [10.0, 1.0])
    action_space_type: str = "surface_agent"


# ------------------------------------------------------------------------------------ #
# Multiple LMs, for voting


@dataclass
class MultiLMMountConfig:
    # Modified from `PatchAndViewFinderMountConfig`
    agent_id: Union[str, None] = "agent_id_0"
    sensor_ids: Union[List[str], None] = field(
        default_factory=lambda: ["patch_0", "patch_1", "view_finder"]
    )
    height: Union[float, None] = 0.0
    position: List[Union[int, float]] = field(default_factory=lambda: [0.0, 1.5, 0.2])
    resolutions: List[List[Union[int, float]]] = field(
        default_factory=lambda: [[64, 64], [64, 64], [64, 64]]
    )
    positions: List[List[Union[int, float]]] = field(
        default_factory=lambda: [
            [0.0, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    rotations: List[List[Union[int, float]]] = field(
        default_factory=lambda: [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    semantics: List[List[Union[int, float]]] = field(
        default_factory=lambda: [False, False, False]
    )
    zooms: List[float] = field(default_factory=lambda: [10.0, 10.0, 1.0])


@dataclass
class TwoLMStackedDistantMountConfig:
    # two sensor patches at the same location with different receptive field sizes
    # Used for basic test with heterarchy.
    agent_id: Union[str, None] = "agent_id_0"
    sensor_ids: Union[List[str], None] = field(
        default_factory=lambda: ["patch_0", "patch_1", "view_finder"]
    )
    height: Union[float, None] = 0.0
    position: List[Union[int, float]] = field(default_factory=lambda: [0.0, 1.5, 0.2])
    resolutions: List[List[Union[int, float]]] = field(
        default_factory=lambda: [[64, 64], [64, 64], [64, 64]]
    )
    positions: List[List[Union[int, float]]] = field(
        default_factory=lambda: [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    rotations: List[List[Union[int, float]]] = field(
        default_factory=lambda: [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    semantics: List[List[Union[int, float]]] = field(
        default_factory=lambda: [False, False, False]
    )
    zooms: List[float] = field(default_factory=lambda: [10.0, 5.0, 1.0])


@dataclass
class TwoLMStackedSurfaceMountConfig(TwoLMStackedDistantMountConfig):
    action_space_type: str = "surface_agent"


@dataclass
class FiveLMMountConfig:
    # Modified from `PatchAndViewFinderMountConfig`
    agent_id: Union[str, None] = "agent_id_0"
    sensor_ids: Union[List[str], None] = field(
        default_factory=lambda: [
            "patch_0",
            "patch_1",
            "patch_2",
            "patch_3",
            "patch_4",
            "view_finder",
        ]
    )
    height: Union[float, None] = 0.0
    position: List[Union[int, float]] = field(default_factory=lambda: [0.0, 1.5, 0.2])
    resolutions: List[List[Union[int, float]]] = field(
        default_factory=lambda: [
            [64, 64],
            [64, 64],
            [64, 64],
            [64, 64],
            [64, 64],
            [64, 64],
        ]
    )
    positions: List[List[Union[int, float]]] = field(
        default_factory=lambda: [
            [0.0, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, -0.01, 0.0],
            [0.01, 0.0, 0.0],
            [-0.01, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    rotations: List[List[Union[int, float]]] = field(
        default_factory=lambda: [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    semantics: List[List[Union[int, float]]] = field(
        default_factory=lambda: [False, False, False, False, False, False]
    )
    zooms: List[float] = field(
        default_factory=lambda: [10.0, 10.0, 10.0, 10.0, 10.0, 1.0]
    )


@dataclass
class PatchAndViewFinderMultiObjectMountConfig(PatchAndViewFinderMountConfig):
    semantics: List[List[Union[int, float]]] = field(
        default_factory=lambda: [True, True]
    )


"""
Utilities for generating multi-LM dataset args.
"""


def make_sensor_positions_on_grid(
    n_sensors: int,
    delta: Number = 0.01,
    order_by: str = "distance",
    add_view_finder: bool = True,
) -> np.ndarray:
    """Generate sensor positions on a 2D grid.

    Create mounting positions for an arbitrary number of sensors, where the
    sensors lie on an imaginary grid on the xy plane (and z = 0). Sensor position
    0 is always centered at (0, 0, 0), and all other sensors are clustered
    around it. The method for selecting which grid points around the center to assign
    each sensor is determined by the `order_by` argument (see below).

    By default, `n_sensors + 1` positions are returned; the first `n_sensors` positions
    are for regular sensors, and an additional position is appended by default to
    accommodate a view finder. The view finder position (if used) is the same as
    sensor position 0 (i.e., (0, 0, 0)).

    Args:
        n_sensors: Number of sensors. Count should not include a view finder.
        delta: The grid spacing length. By default, sensors will be
            placed every centimeter (units are in meters).
        order_by: How to select points on the grid that will contain
            sensors.
             - "spiral": sensors are numbered along a counter-clockwise spiral
                spreading outwards from the center.
             - "distance": sensors are ordered by their distance from the center.
                This can result in a more jagged pattern along the edges but
                results in sensors generally more packed towards the center.
                Positions that are equidistant from the center are ordered
                counterclockwise starting at 3 o'clock.
        add_view_finder: Whether to include an extra position module
            at the origin to serve as a view finder. Defaults to `True`.

    Returns:
        A 2D array of sensor positions where each row is an array of (x, y, z)
        positions. If `add_view_finder` is True, the array has `n_sensors + 1` rows,
        where the last row corresponds to the view finder's position and is identical to
        row 0. Otherwise, the array has `n_sensors` rows. row 0 is always centered at
        (0, 0, 0), and all other rows are offset relative to it.

    """
    assert n_sensors > 0, "n_sensors must be greater than 0"
    assert delta > 0, "delta must be greater than 0"
    assert order_by in ["spiral", "distance"], "order_by must be 'spiral' or 'distance'"

    # Find smallest square grid size that can fit n_lms with odd-length sides.
    grid_size = 1
    while n_sensors > grid_size**2:
        grid_size += 2

    # Make coordinate grids, where the center is (0, 0).
    points = np.arange(-grid_size // 2 + 1, grid_size // 2 + 1)
    x, y = np.meshgrid(points, points)
    y = np.flipud(y)  # Flip y-axis to match habitat coordinate system (positive is up).
    i_mid = grid_size // 2

    if order_by == "distance":
        dists = x**2 + y**2
        unique_dists = np.sort(np.unique(dists))
        assert unique_dists[0] == 0
        indices = []
        for i in range(len(unique_dists)):
            u = unique_dists[i]
            inds = np.argwhere(dists == u)
            angles = np.arctan2(i_mid - inds[:, 1], inds[:, 0] - i_mid)
            sorting_inds = np.argsort(angles)
            inds = inds[sorting_inds]
            indices.extend(list(inds))

    elif order_by == "spiral":
        indices = [(i_mid, i_mid)]

        # Directions for moving in spiral: right, down, left, up
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        current_dir = 0  # Start moving right
        steps = 1  # How many steps to take in current direction
        steps_taken = 0  # Steps taken in current direction
        row, col = i_mid, i_mid  # Start at center

        # Generate spiral pattern until we have enough points
        while len(indices) < grid_size**2:
            # Move in current direction
            row += directions[current_dir][0]
            col += directions[current_dir][1]

            # Add point if it's within bounds
            if 0 <= row < grid_size and 0 <= col < grid_size:
                indices.append((row, col))

            steps_taken += 1

            # Check if we need to change direction
            if steps_taken == steps:
                steps_taken = 0
                current_dir = (current_dir + 1) % 4
                # Increase steps every 2 direction changes (completing half circle)
                if current_dir % 2 == 0:
                    steps += 1

    indices = np.array(indices)[:n_sensors]

    # Convert indices to locations in agent space.
    positions = []
    for idx in indices:
        positions.append((x[idx[0], idx[1]] * delta, y[idx[0], idx[1]] * delta))
    positions = np.array(positions)

    # Add z-positions.
    positions = np.hstack((positions, np.zeros([positions.shape[0], 1])))

    # Optionally append entry for a view finder which is a duplicate of row zero.
    # Should be (0, 0, 0).
    if add_view_finder:
        positions = np.vstack([positions, positions[0].reshape(1, -1)])

    return positions


def make_multi_sensor_mount_config(
    n_sensors: int,
    agent_id: str = "agent_id_0",
    sensor_ids: Optional[Sequence[str]] = None,
    height: Number = 0.0,
    position: ArrayLike = (0, 1.5, 0.2),  # agent position
    resolutions: Optional[ArrayLike] = None,
    positions: Optional[ArrayLike] = None,
    rotations: Optional[ArrayLike] = None,
    semantics: Optional[ArrayLike] = None,
    zooms: Optional[ArrayLike] = None,
) -> Mapping[str, Any]:
    """Generate a multi-sensor mount configuration.

    Creates a multi-sensor, single-agent mount config. Its primary use is in generating
    a `MultiLMMountHabitatDatasetArgs` config. Defaults are reasonable and reflect
    current common practices.

    Note:
        `n_sensors` indicates the number of non-view-finder sensors. However, the
        arrays generated for `sensor_ids`, `resolutions`, `positions`, `rotations`,
        `semantics`, and `zooms` will have length `n_sensors + 1` to accommodate a
        view finder. As such, arguments supplied for these arrays must also have length
        `n_sensors + 1`, where the view finder's values come last.

    Args:
        n_sensors: Number of sensors, not including the view finder.
        agent_id: ID of the agent. Defaults to "agent_id_0".
        sensor_ids: IDs of the sensor modules. Defaults to
            `["patch_0", "patch_1", ... "patch_{n_sms - 1}", "view_finder"]`.
        height: Height of the agent. Defaults to 0.0.
        position: Position of the agent. Defaults to [0, 1.5, 0.2].
        resolutions: Resolutions of the sensors. Defaults to (64, 64) for all
            sensors.
        positions: Positions of the sensors. If not provided, calls
            `make_sensor_positions_on_grid` with its default arguments.
        rotations: Rotations of the sensors. Defaults to [1, 0, 0, 0] for all sensors.
        semantics: Defaults to `False` for all sensors.
        zooms: Zooms of the sensors. Defaults to 10.0 for all sensors except for the
          except for the view finder (which has a zoom of 1.0)

    Returns:
        A dictionary representing a complete multi-sensor mount config. Arrays are
        converted to lists.

    """
    assert n_sensors > 0, "n_sensors must be a positive integer"
    arr_len = n_sensors + 1

    # Initialize with agent info, then add sensor info.
    mount_config = {
        "agent_id": str(agent_id),
        "height": float(height),
        "position": np.asarray(position),
    }

    # sensor IDs.
    if sensor_ids is None:
        sensor_ids = np.array(
            [f"patch_{i}" for i in range(n_sensors)] + ["view_finder"], dtype=object
        )
    else:
        sensor_ids = np.array(sensor_ids, dtype=object)
    assert sensor_ids.shape == (arr_len,), f"`sensor_ids` must have length {arr_len}"
    mount_config["sensor_ids"] = sensor_ids

    # sensor resolutions
    if resolutions is None:
        resolutions = np.full([arr_len, 2], 64)
    else:
        resolutions = np.asarray(resolutions)
    assert resolutions.shape == (
        arr_len,
        2,
    ), f"`resolutions` must have shape ({arr_len}, 2)"
    mount_config["resolutions"] = resolutions

    # sensor positions
    if positions is None:
        positions = make_sensor_positions_on_grid(
            n_sensors=n_sensors,
            add_view_finder=True,
        )
    else:
        positions = np.asarray(positions)
    assert positions.shape == (
        arr_len,
        3,
    ), f"`positions` must have shape ({arr_len}, 3)"
    mount_config["positions"] = positions

    # sensor rotations
    if rotations is None:
        rotations = np.zeros([arr_len, 4])
        rotations[:, 0] = 1.0
    else:
        rotations = np.asarray(rotations)
    assert rotations.shape == (
        arr_len,
        4,
    ), f"`rotations` must have shape ({arr_len}, 4)"
    mount_config["rotations"] = rotations

    # sensor semantics
    if semantics is None:
        semantics = np.zeros(arr_len, dtype=bool)
    else:
        semantics = np.asarray(semantics, dtype=bool)
    assert semantics.shape == (arr_len,), f"`semantics` must have shape ({arr_len},)"
    mount_config["semantics"] = semantics

    # sensor zooms
    if zooms is None:
        zooms = 10.0 * np.ones(arr_len)
        zooms[-1] = 1.0  # view finder
    else:
        zooms = np.asarray(zooms)
    assert zooms.shape == (arr_len,), f"`zooms` must have shape ({arr_len},)"
    mount_config["zooms"] = zooms

    return mount_config
