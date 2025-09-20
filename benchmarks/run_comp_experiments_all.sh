#!/usr/bin/env bash

python run.py -e supervised_pre_training_flat_objects_wo_logos \
    supervised_pre_training_logos_after_flat_objects \
    supervised_pre_training_curved_objects_after_flat_and_logo \
    supervised_pre_training_objects_with_logos_lvl1_monolithic_models \
    supervised_pre_training_objects_with_logos_lvl1_comp_models \
    supervised_pre_training_objects_with_logos_lvl2_comp_models \
    supervised_pre_training_objects_with_logos_lvl3_comp_models \
    supervised_pre_training_objects_with_logos_lvl4_comp_models \
    infer_comp_lvl1_with_monolithic_models \
    infer_parts_with_part_models \
    infer_comp_lvl1_with_comp_models \
    infer_comp_lvl2_with_comp_models \
    infer_comp_lvl3_with_comp_models \
    infer_comp_lvl4_with_comp_models
