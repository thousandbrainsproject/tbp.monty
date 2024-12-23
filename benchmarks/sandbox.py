from tbp.monty.frameworks.run_env import setup_env

setup_env()
from configs import CONFIGS  # noqa: E402

from tbp.monty.frameworks.run import main  # noqa: E402

main(all_configs=CONFIGS, experiments=["base_config_10distinctobj_surf_agent"])

# import matplotlib.pyplot as plt
# import numpy as np

# from tbp.monty.frameworks.environment_utils.transforms import DepthTo3DLocations

# t = DepthTo3DLocations(
#     agent_id="agent_id_0",
#     sensor_ids=["patch", "view_finder"],
#     resolutions=[[64, 64], [64, 64]],
#     zooms=[1.0, 10.0],
#     get_all_points=True,
# )

# depth = np.ones((64, 64))
# depth[32, 32] = 0

# observations = {
#     "agent_id_0": {
#         "patch": {
#             "depth": np.random.rand(64, 64),
#             "rgba": np.random.rand(64, 64, 4),
#         },
#         "view_finder": {
#             "depth": depth,
#             "rgba": np.random.rand(64, 64, 4),
#         },
#     },
# }

# obs = t(observations)
# depth = obs["agent_id_0"]["view_finder"]["depth"]
# sem = obs["agent_id_0"]["view_finder"]["semantic_3d"][:, 3].reshape(depth.shape)


# plt.imshow(sem)
# plt.colorbar()
# plt.show()
