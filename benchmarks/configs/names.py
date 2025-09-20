# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""The names of declared experiments grouped by category.

This module keeps experiment names separate from the configuration for the
experiments. The reason for doing this is so that we can import the configurations
selectively to avoid importing uninstalled dependencies (e.g., not installing a
specific simulator).

The use of dataclasses assists in raising early errors when experiment names defined
here and the corresponding experiment configurations drift apart. For additional
discussion, see: https://github.com/thousandbrainsproject/tbp.monty/pull/153.
"""

import inspect
import sys
from dataclasses import dataclass, fields, is_dataclass

from benchmarks.configs.follow_ups.names import NAMES as FOLLOW_UP_NAMES

NAMES = []
NAMES.extend(FOLLOW_UP_NAMES)


@dataclass
class MontyWorldExperiments:
    world_image_from_stream_on_scanned_model: dict
    world_image_on_scanned_model: dict
    dark_world_image_on_scanned_model: dict
    bright_world_image_on_scanned_model: dict
    hand_intrusion_world_image_on_scanned_model: dict
    multi_object_world_image_on_scanned_model: dict


@dataclass
class MontyWorldHabitatExperiments:
    randrot_noise_sim_on_scan_monty_world: dict


@dataclass
class PretrainingExperiments:
    supervised_pre_training_base: dict
    supervised_pre_training_5lms: dict
    supervised_pre_training_5lms_all_objects: dict
    only_surf_agent_training_10obj: dict
    only_surf_agent_training_10simobj: dict
    only_surf_agent_training_allobj: dict
    only_surf_agent_training_numenta_lab_obj: dict


@dataclass
class CompositionalInferenceExperiments:
    infer_comp_lvl1_with_monolithic_models: dict
    infer_parts_with_part_models: dict
    infer_comp_lvl1_with_comp_models: dict
    infer_comp_lvl2_with_comp_models: dict
    infer_comp_lvl3_with_comp_models: dict
    infer_comp_lvl4_with_comp_models: dict


@dataclass
class CompositionalLearningExperiments:
    supervised_pre_training_flat_objects_wo_logos: dict
    supervised_pre_training_logos_after_flat_objects: dict
    supervised_pre_training_curved_objects_after_flat_and_logo: dict
    supervised_pre_training_objects_with_logos_lvl1_monolithic_models: dict
    supervised_pre_training_objects_with_logos_lvl1_comp_models: dict
    supervised_pre_training_objects_with_logos_lvl2_comp_models: dict
    supervised_pre_training_objects_with_logos_lvl3_comp_models: dict
    supervised_pre_training_objects_with_logos_lvl4_comp_models: dict


@dataclass
class YcbExperiments:
    base_config_10distinctobj_dist_agent: dict
    base_config_10distinctobj_surf_agent: dict
    randrot_noise_10distinctobj_dist_agent: dict
    randrot_noise_10distinctobj_dist_on_distm: dict
    randrot_noise_10distinctobj_surf_agent: dict
    randrot_10distinctobj_surf_agent: dict
    randrot_noise_10distinctobj_5lms_dist_agent: dict
    base_10simobj_surf_agent: dict
    randrot_noise_10simobj_surf_agent: dict
    randrot_noise_10simobj_dist_agent: dict
    randomrot_rawnoise_10distinctobj_surf_agent: dict
    base_10multi_distinctobj_dist_agent: dict
    surf_agent_unsupervised_10distinctobj: dict
    surf_agent_unsupervised_10distinctobj_noise: dict
    surf_agent_unsupervised_10simobj: dict
    base_77obj_dist_agent: dict
    base_77obj_surf_agent: dict
    randrot_noise_77obj_surf_agent: dict
    randrot_noise_77obj_dist_agent: dict
    randrot_noise_77obj_5lms_dist_agent: dict


@dataclass
class UnsupervisedInferenceExperiments:
    unsupervised_inference_distinctobj_surf_agent: dict
    unsupervised_inference_distinctobj_dist_agent: dict


@dataclass
class MyExperiments:
    # Add your experiment names here
    pass


current_module = sys.modules[__name__]
for _name, obj in inspect.getmembers(current_module):
    if inspect.isclass(obj) and is_dataclass(obj):
        NAMES.extend(f.name for f in fields(obj))
