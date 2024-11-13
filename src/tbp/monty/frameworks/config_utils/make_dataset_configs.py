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
from typing import Callable, Dict, List, Union

import numpy as np
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.environment_utils.transforms import (
    AddNoiseToRawDepthImage,
    DepthTo3DLocations,
    MissingToMaxDepth,
)
from tbp.monty.frameworks.environments.habitat import (
    AgentConfig,
    HabitatEnvironment,
    ObjectConfig,
)
from tbp.monty.frameworks.environments.two_d_data import (
    OmniglotEnvironment,
    SaccadeOnImageEnvironment,
    SaccadeOnImageFromStreamEnvironment,
)
from tbp.monty.frameworks.environments.ycb import SHUFFLED_YCB_OBJECTS
from tbp.monty.frameworks.utils.transform_utils import scipy_to_numpy_quat
from tbp.monty.simulators.habitat import MultiSensorAgent, SingleSensorAgent

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


@dataclass
class EnvInitArgs:
    """Args for :class:`HabitatEnvironment`."""

    agents: List[AgentConfig]
    objects: List[ObjectConfig] = field(
        default_factory=lambda: [ObjectConfig("coneSolid", position=(0.0, 1.5, -0.1))]
    )
    scene_id: Union[int, None] = field(default=None)
    seed: int = field(default=42)
    data_path: str = os.path.expanduser("~/tbp/data/habitat/objects/ycb")


@dataclass
class EnvInitArgsSinglePTZ(EnvInitArgs):
    """Use this to make a sim with a cone and a single PTZCameraAgent."""

    agents: List[AgentConfig] = field(
        default_factory=lambda: [
            AgentConfig(
                SingleSensorAgent,
                dict(
                    agent_id="agent_id_0",
                    sensor_id="sensor_id_0",
                    resolution=(320, 240),
                ),
            )
        ]
    )


@dataclass
class EnvInitArgsSimpleMount(EnvInitArgs):
    """Use this to make a sim with a cone and a single mount agent with two cameras."""

    agents: List[AgentConfig] = field(
        default_factory=lambda: [
            AgentConfig(MultiSensorAgent, TwoCameraMountConfig().__dict__)
        ]
    )


@dataclass
class EnvInitArgsPatchViewMount(EnvInitArgs):
    agents: List[AgentConfig] = field(
        default_factory=lambda: [
            AgentConfig(MultiSensorAgent, PatchAndViewFinderMountConfig().__dict__)
        ]
    )


@dataclass
class EnvInitArgsShapenetPatchViewMount(EnvInitArgsPatchViewMount):
    data_path: str = os.path.expanduser("~/tbp/data/shapenet")


@dataclass
class EnvInitArgsMontyWorldPatchViewMount(EnvInitArgsPatchViewMount):
    data_path: str = os.path.expanduser("~/tbp/data/numenta_lab")


# Data-set containing RGBD images of real-world objects taken with a mobile device
@dataclass
class EnvInitArgsMontyWorldStandardScenes:
    data_path: str = os.path.expanduser("~/tbp/data/worldimages/standard_scenes/")


@dataclass
class EnvInitArgsMontyWorldBrightScenes:
    data_path: str = os.path.expanduser("~/tbp/data/worldimages/bright_scenes/")


@dataclass
class EnvInitArgsMontyWorldDarkScenes:
    data_path: str = os.path.expanduser("~/tbp/data/worldimages/dark_scenes/")


# Data-set where a hand is prominently visible holding (and thereby partially
# occluding) the objects
@dataclass
class EnvInitArgsMontyWorldHandIntrusionScenes:
    data_path: str = os.path.expanduser("~/tbp/data/worldimages/hand_intrusion_scenes/")


# Data-set where there are two objects in the image; the target class is in the centre
# of the image; of the 4 images per target, the first two images contain similar objects
# (e.g. another type of mug), while the last two images contain distinct objects (such
# as a book if the target is a type of mug)
@dataclass
class EnvInitArgsMontyWorldMultiObjectScenes:
    data_path: str = os.path.expanduser("~/tbp/data/worldimages/multi_object_scenes/")


@dataclass
class EnvInitArgsSurfaceViewMount(EnvInitArgs):
    agents: List[AgentConfig] = field(
        default_factory=lambda: [
            AgentConfig(MultiSensorAgent, SurfaceAndViewFinderMountConfig().__dict__)
        ]
    )


@dataclass
class EnvInitArgsMontyWorldSurfaceViewMount(EnvInitArgsMontyWorldPatchViewMount):
    agents: List[AgentConfig] = field(
        default_factory=lambda: [
            AgentConfig(MultiSensorAgent, SurfaceAndViewFinderMountConfig().__dict__)
        ]
    )


@dataclass
class EnvInitArgsPatchViewMountLowRes(EnvInitArgs):
    agents: List[AgentConfig] = field(
        default_factory=lambda: [
            AgentConfig(
                MultiSensorAgent, PatchAndViewFinderMountLowResConfig().__dict__
            )
        ]
    )


@dataclass
class SinglePTZHabitatDatasetArgs:
    """Define a dataset with a single cone and a single PTZCameraAgent.

    Use this to make a :class:`EnvironmentDataset` with an env with a single cone and
    a single PTZCameraAgent.
    """

    env_init_func: Callable = field(default=HabitatEnvironment)
    env_init_args: Union[Dict, dataclass] = field(
        default_factory=lambda: EnvInitArgsSinglePTZ().__dict__
    )
    transform: Union[Callable, list, None] = field(default=None)


@dataclass
class SimpleMountHabitatDatasetArgs:
    """Define a dataset with a single cone and a single mount agent with two cameras.

    Use this to make a :class:`EnvironmentDataset` with an env with a single cone and
    a single mount agent with two cameras.
    """

    env_init_func: Callable = field(default=HabitatEnvironment)
    env_init_args: Dict = field(
        default_factory=lambda: EnvInitArgsSimpleMount().__dict__
    )
    transform: Union[Callable, list, None] = field(default=None)


@dataclass
class PatchViewFinderMountHabitatDatasetArgs:
    env_init_func: Callable = field(default=HabitatEnvironment)
    env_init_args: Dict = field(
        default_factory=lambda: EnvInitArgsPatchViewMount().__dict__
    )
    transform: Union[Callable, list, None] = None
    rng: Union[Callable, None] = None

    def __post_init__(self):
        agent_args = self.env_init_args["agents"][0].agent_args
        self.transform = [
            MissingToMaxDepth(agent_id=agent_args["agent_id"], max_depth=1),
            DepthTo3DLocations(
                agent_id=agent_args["agent_id"],
                sensor_ids=agent_args["sensor_ids"],
                resolutions=agent_args["resolutions"],
                world_coord=True,
                zooms=agent_args["zooms"],
                get_all_points=True,
                use_semantic_sensor=True,
            ),
        ]


@dataclass
class NoisyPatchViewFinderMountHabitatDatasetArgs:
    env_init_func: Callable = field(default=HabitatEnvironment)
    env_init_args: Dict = field(
        default_factory=lambda: EnvInitArgsPatchViewMount().__dict__
    )
    transform: Union[Callable, list, None] = None

    def __post_init__(self):
        agent_args = self.env_init_args["agents"][0].agent_args
        self.transform = [
            MissingToMaxDepth(agent_id=agent_args["agent_id"], max_depth=1),
            AddNoiseToRawDepthImage(
                agent_id=agent_args["agent_id"],
                sigma=0.001,
            ),  # add gaussian noise with 0.001 std to depth image
            # Uncomment line below to enable smoothing of sensor patch depth
            # GaussianSmoothing(agent_id=agent_args["agent_id"]),
            DepthTo3DLocations(
                agent_id=agent_args["agent_id"],
                sensor_ids=agent_args["sensor_ids"],
                resolutions=agent_args["resolutions"],
                world_coord=True,
                zooms=agent_args["zooms"],
                get_all_points=True,
                use_semantic_sensor=True,
            ),
        ]


@dataclass
class OmniglotDatasetArgs:
    env_init_func: Callable = field(default=OmniglotEnvironment)
    env_init_args: Dict = field(default_factory=lambda: dict())
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
                use_semantic_sensor=True,
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
    env_init_args: Dict = field(default_factory=lambda: dict())
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
class PatchViewFinderLowResMountHabitatDatasetArgs(
    PatchViewFinderMountHabitatDatasetArgs
):
    env_init_args: Dict = field(
        default_factory=lambda: EnvInitArgsPatchViewMountLowRes().__dict__
    )


@dataclass
class PatchViewFinderShapenetMountHabitatDatasetArgs(
    PatchViewFinderMountHabitatDatasetArgs
):
    env_init_args: Dict = field(
        default_factory=lambda: EnvInitArgsShapenetPatchViewMount().__dict__
    )


@dataclass
class PatchViewFinderMontyWorldMountHabitatDatasetArgs(
    PatchViewFinderMountHabitatDatasetArgs
):
    env_init_args: Dict = field(
        default_factory=lambda: EnvInitArgsMontyWorldPatchViewMount().__dict__
    )


@dataclass
class SurfaceViewFinderMountHabitatDatasetArgs(PatchViewFinderMountHabitatDatasetArgs):
    env_init_args: Dict = field(
        default_factory=lambda: EnvInitArgsSurfaceViewMount().__dict__
    )

    def __post_init__(self):
        agent_args = self.env_init_args["agents"][0].agent_args
        self.transform = [
            MissingToMaxDepth(agent_id=agent_args["agent_id"], max_depth=1),
            DepthTo3DLocations(
                agent_id=agent_args["agent_id"],
                sensor_ids=agent_args["sensor_ids"],
                resolutions=agent_args["resolutions"],
                world_coord=True,
                zooms=agent_args["zooms"],
                get_all_points=True,
                use_semantic_sensor=True,
                depth_clip_sensors=(0,),  # comma needed to make it a tuple
                clip_value=0.05,
            ),
        ]


@dataclass
class SurfaceViewFinderMontyWorldMountHabitatDatasetArgs(
    SurfaceViewFinderMountHabitatDatasetArgs
):
    env_init_args: Dict = field(
        default_factory=lambda: EnvInitArgsMontyWorldSurfaceViewMount().__dict__
    )


@dataclass
class NoisySurfaceViewFinderMountHabitatDatasetArgs(
    PatchViewFinderMountHabitatDatasetArgs
):
    env_init_args: Dict = field(
        default_factory=lambda: EnvInitArgsSurfaceViewMount().__dict__
    )

    def __post_init__(self):
        agent_args = self.env_init_args["agents"][0].agent_args
        self.transform = [
            MissingToMaxDepth(agent_id=agent_args["agent_id"], max_depth=1),
            AddNoiseToRawDepthImage(
                agent_id=agent_args["agent_id"],
                sigma=0.001,
            ),  # add gaussian noise with 0.001 std to depth image
            # Uncomment line below to enable smoothing of sensor patch depth
            # GaussianSmoothing(agent_id=agent_args["agent_id"]),
            DepthTo3DLocations(
                agent_id=agent_args["agent_id"],
                sensor_ids=agent_args["sensor_ids"],
                resolutions=agent_args["resolutions"],
                world_coord=True,
                zooms=agent_args["zooms"],
                get_all_points=True,
                use_semantic_sensor=True,
                depth_clip_sensors=(0,),  # comma needed to make it a tuple
                clip_value=0.05,
            ),
        ]


@dataclass
class ObjectList:
    """Use this to make a list of :class:`ObjectConfig` s.

    See Also:
        tbp.monty.frameworks.environments.habitat ObjectConfig
    """

    object_names: List[str]

    def __post_init__(self):
        self.object_configs = [ObjectConfig(name) for name in self.object_names]


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
        data_path = os.path.expanduser("~/tbp/data/omniglot/python/")
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
        characters_in_a = [
            c for c in os.listdir(data_path + "images_background/" + alphabet)
        ]
        for c_idx, character in enumerate(characters_in_a):
            versions_of_char = [
                v
                for v in os.listdir(
                    data_path + "images_background/" + alphabet + "/" + character
                )
            ]
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
        data_path = os.path.expanduser("~/tbp/data/omniglot/python/")
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
        characters_in_a = [
            c for c in os.listdir(data_path + "images_background/" + alphabet)
        ]
        for c_idx, character in enumerate(characters_in_a):
            if num_versions is None:
                versions_of_char = [
                    v
                    for v in os.listdir(
                        data_path + "images_background/" + alphabet + "/" + character
                    )
                ]
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
        default_factory=lambda: [True, True]
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
class MultiLMMountHabitatDatasetArgs:
    env_init_func: Callable = field(default=HabitatEnvironment)
    env_init_args: Dict = field(
        default_factory=lambda: EnvInitArgsMultiLMMount().__dict__
    )
    transform: Union[Callable, list, None] = None

    def __post_init__(self):
        agent_args = self.env_init_args["agents"][0].agent_args
        self.transform = [
            MissingToMaxDepth(agent_id=agent_args["agent_id"], max_depth=1),
            DepthTo3DLocations(
                agent_id=agent_args["agent_id"],
                sensor_ids=agent_args["sensor_ids"],
                resolutions=agent_args["resolutions"],
                world_coord=True,
                zooms=agent_args["zooms"],
                get_all_points=True,
                use_semantic_sensor=True,
            ),
        ]


@dataclass
class TwoLMStackedDistantMountHabitatDatasetArgs(MultiLMMountHabitatDatasetArgs):
    env_init_args: Dict = field(
        default_factory=lambda: EnvInitArgsTwoLMDistantStackedMount().__dict__
    )


@dataclass
class TwoLMStackedSurfaceMountHabitatDatasetArgs(MultiLMMountHabitatDatasetArgs):
    env_init_args: Dict = field(
        default_factory=lambda: EnvInitArgsTwoLMSurfaceStackedMount().__dict__
    )


@dataclass
class FiveLMMountHabitatDatasetArgs(MultiLMMountHabitatDatasetArgs):
    env_init_args: Dict = field(
        default_factory=lambda: EnvInitArgsFiveLMMount().__dict__
    )


@dataclass
class EnvInitArgsMultiLMMount(EnvInitArgs):
    agents: List[AgentConfig] = field(
        default_factory=lambda: [
            AgentConfig(MultiSensorAgent, MultiLMMountConfig().__dict__)
        ]
    )


@dataclass
class EnvInitArgsTwoLMDistantStackedMount(EnvInitArgs):
    agents: List[AgentConfig] = field(
        default_factory=lambda: [
            AgentConfig(MultiSensorAgent, TwoLMStackedDistantMountConfig().__dict__)
        ]
    )


@dataclass
class EnvInitArgsTwoLMSurfaceStackedMount(EnvInitArgs):
    agents: List[AgentConfig] = field(
        default_factory=lambda: [
            AgentConfig(MultiSensorAgent, TwoLMStackedSurfaceMountConfig().__dict__)
        ]
    )


@dataclass
class EnvInitArgsFiveLMMount(EnvInitArgs):
    agents: List[AgentConfig] = field(
        default_factory=lambda: [
            AgentConfig(MultiSensorAgent, FiveLMMountConfig().__dict__)
        ]
    )


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
        default_factory=lambda: [True, True, True]
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
        default_factory=lambda: [True, True, True]
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
        default_factory=lambda: [True, True, True, True, True, True]
    )
    zooms: List[float] = field(
        default_factory=lambda: [10.0, 10.0, 10.0, 10.0, 10.0, 1.0]
    )
