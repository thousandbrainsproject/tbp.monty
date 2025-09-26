#!/usr/bin/env bash

python run.py -e infer_comp_lvl1_with_monolithic_models \
    infer_parts_with_part_models \
    infer_comp_lvl1_with_comp_models \
    infer_comp_lvl2_with_comp_models \
    infer_comp_lvl3_with_comp_models \
    infer_comp_lvl4_with_comp_models
