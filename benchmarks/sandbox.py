from tbp.monty.frameworks.run_env import setup_env

setup_env()
from configs import CONFIGS  # noqa: E402

from tbp.monty.frameworks.run import main  # noqa: E402

main(all_configs=CONFIGS, experiments=["base_config_10distinctobj_surf_agent"])

"""
Unsupervised
----------------------------------------------------------------------------------------
"""

# from tbp.monty.frameworks.utils.logging_utils import (
#     load_stats,
#     print_unsupervised_stats,
# )

# exp_path = "/Users/sknudstrup/tbp/results/monty/projects/monty_runs/surf_agent_unsupervised_10distinctobj"
# stats, _, _, _ = load_stats(
#     exp_path, load_eval=False, load_detailed=False, load_models=False
# )
# print_unsupervised_stats(stats, 10)
