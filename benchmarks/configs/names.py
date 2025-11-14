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


@dataclass
class PretrainingExperiments:
    supervised_pre_training_base: dict


@dataclass
class CompositionalLearningExperiments:
    supervised_pre_training_flat_objects_wo_logos: dict
    supervised_pre_training_logos_after_flat_objects: dict
    supervised_pre_training_curved_objects_after_flat_and_logo: dict
    supervised_pre_training_objects_with_logos_lvl1_monolithic_models: dict
    supervised_pre_training_objects_with_logos_lvl1_comp_models: dict
    supervised_pre_training_objects_with_logos_lvl1_comp_models_resampling: dict
    supervised_pre_training_objects_with_logos_lvl2_comp_models: dict
    supervised_pre_training_objects_with_logos_lvl3_comp_models: dict
    supervised_pre_training_objects_with_logos_lvl4_comp_models: dict


class MyExperiments:
    # Add your experiment names here
    pass


current_module = sys.modules[__name__]
for _name, obj in inspect.getmembers(current_module):
    if inspect.isclass(obj) and is_dataclass(obj):
        NAMES.extend(f.name for f in fields(obj))
