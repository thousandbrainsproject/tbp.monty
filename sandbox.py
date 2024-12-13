# %%
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tbp.monty.frameworks.run import main

# module_path = (
#     "/Users/sknudstrup/tbp/monty_lab/dmc_configs/dmc_configs/patch_off_evals.py"
# )
# experiment_name = "dist_agent_1lm"

module_path = Path(
    "/Users/sknudstrup/tbp/tbp.monty/benchmarks/configs/ycb_experiments.py"
)
experiment_name = "randrot_noise_77obj_dist_agent_test"

module_name = module_path.stem
module_path = Path(module_path).expanduser()
module_name = module_path.stem
spec = importlib.util.spec_from_file_location(module_name, module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

main(module.CONFIGS, [experiment_name])
