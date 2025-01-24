# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from dataclasses import dataclass


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
