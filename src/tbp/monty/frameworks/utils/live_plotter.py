# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.abstract_monty_classes import Monty, Observations
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.frameworks.utils.plot_utils import add_patch_outline_to_view_finder

# turn interactive plotting off -- call plt.show() to open all figures
plt.ioff()


class LivePlotter:
    """Class for plotting sensor observations during an experiment.

    Set the `show_sensor_output` flag in the experiment config to True to enable live
    plotting.

    WARNING: This plotter makes a number of assumptions right now. For example, it
    assumes that
    - sensor with ID "view_finder" exists
    - sensor with ID "patch" exists
    - an image modality exists in both sensor observations
    """

    def __init__(self):
        pass

    def initialize_online_plotting(self):
        self.fig, self.ax = plt.subplots(1, 3, figsize=(9, 6))
        self.fig.subplots_adjust(top=1.1)
        # self.colorbar = self.fig.colorbar(None, fraction=0.046, pad=0.04)
        self.setup_camera_ax()
        self.setup_sensor_ax()
        self.setup_mlh_ax()

    def hardcoded_assumptions(self, observation: Observations, model: Monty):
        """Extract some of the hardcoded assumptions from the observation.

        TODO: Don't do this. It is here for now to highlight the fragility of the
        live plotter implementation at the call site. We should make this less
        fragile by passing the necessary information to the live plotter.

        Args:
            observation: The observation from the environment interface.
            model: The model.

        Returns:
            A tuple of the first learning module, the first sensor module raw
            observations, the patch image, and the view finder image.

        Raises:
            KeyError: If either sensor observation has no supported image modality.
        """
        first_learning_module = model.learning_modules[0]
        first_sensor_module = model.sensor_modules[0]
        snapshot_telemetry = getattr(first_sensor_module, "_snapshot_telemetry", None)
        first_sensor_module_raw_observations = (
            snapshot_telemetry.raw_observations
            if snapshot_telemetry is not None
            else []
        )
        first_sensor_module_id = first_sensor_module.sensor_module_id

        # Find agent_id corresponding to the first_sensor_module_id
        first_sensor_module_agent_id: AgentID | None = None
        for agent_id, agent_observations in observation.items():
            if first_sensor_module_id in agent_observations:
                first_sensor_module_agent_id = agent_id
                break
        assert first_sensor_module_agent_id is not None

        patch_observation = observation[first_sensor_module_agent_id][
            first_sensor_module_id
        ]
        patch_image = patch_observation.get("depth")
        if patch_image is None:
            patch_image = patch_observation.get("rgba")
        if patch_image is None:
            patch_image = patch_observation.get("raw")
        if patch_image is None:
            raise KeyError("Patch observation has no depth, rgba, or raw image")

        view_finder_observation = observation[first_sensor_module_agent_id][
            SensorID("view_finder")
        ]
        view_finder_image = view_finder_observation.get("rgba")
        if view_finder_image is None:
            view_finder_image = view_finder_observation.get("raw")
        if view_finder_image is None:
            raise KeyError("View-finder observation has no rgba or raw image")
        if hasattr(first_learning_module, "get_current_mlh"):
            mlh = first_learning_module.get_current_mlh()
            if mlh["graph_id"] == "no_observations_yet":
                mlh_model = None
            else:
                mlh_model = first_learning_module.graph_memory.get_graph(
                    mlh["graph_id"]
                )[first_sensor_module_id]
        else:
            mlh = None
            mlh_model = None
        return (
            first_learning_module,
            first_sensor_module_raw_observations,
            patch_image,
            view_finder_image,
            mlh,
            mlh_model,
        )

    def show_observations(
        self,
        first_learning_module,
        first_sensor_module_raw_observations,
        patch_image,
        view_finder_image,
        mlh,
        mlh_model,
        step: int,
        is_saccade_on_image_data_loader=False,
    ) -> None:
        self.fig.suptitle(f"Observation at step {step}")
        self.show_view_finder(
            first_sensor_module_raw_observations,
            first_learning_module,
            patch_image,
            view_finder_image,
            is_saccade_on_image_data_loader,
        )
        self.show_patch(patch_image)
        if mlh_model:
            self.show_mlh(mlh, mlh_model)
        plt.pause(0.00001)

    def show_view_finder(
        self,
        first_sensor_module_raw_observations,
        first_learning_module,
        patch_image,
        view_finder_image,
        is_saccade_on_image_data_loader,
    ):
        if self.camera_image:
            self.camera_image.remove()

        if is_saccade_on_image_data_loader:
            center_pixel_id = np.array([200, 200])
            patch_size = np.asarray(patch_image).shape[0]
            raw_obs = first_sensor_module_raw_observations
            if len(raw_obs) > 0:
                center_pixel_id = np.array(raw_obs[-1]["pixel_loc"])
                view_finder_image = add_patch_outline_to_view_finder(
                    view_finder_image, center_pixel_id, patch_size
                )
            self.camera_image = self.ax[0].imshow(view_finder_image, zorder=-99)
        else:
            self.camera_image = self.ax[0].imshow(
                view_finder_image,
                zorder=-99,
            )
            # Show a square in the middle as a rough estimate of where the patch is
            # Note: This isn't exactly the size that the patch actually is.
            image_shape = view_finder_image.shape
            square = plt.Rectangle(
                (image_shape[1] * 4.5 // 10, image_shape[0] * 4.5 // 10),
                image_shape[1] / 10,
                image_shape[0] / 10,
                fc="none",
                ec="white",
            )
            self.ax[0].add_patch(square)
        if hasattr(first_learning_module, "get_current_mlh"):
            mlh = first_learning_module.get_current_mlh()
            if mlh and mlh["graph_id"] != "no_observations_yet":
                graph_ids, evidences = first_learning_module.evidence_for_each_graph()
                self.add_text(
                    mlh,
                    pos=view_finder_image.shape[0],
                    possible_matches=first_learning_module.get_possible_matches(),
                    graph_ids=graph_ids,
                    evidences=evidences,
                )

    def show_patch(self, patch_image):
        if self.sensor_image:
            self.sensor_image.remove()
        kwargs = {"cmap": "viridis_r"} if np.asarray(patch_image).ndim == 2 else {}
        self.sensor_image = self.ax[1].imshow(patch_image, **kwargs)

    def show_mlh(self, mlh, mlh_model):
        if not mlh_model:
            self.ax[2].set_title("No MLH")
            return

        self.ax[2].cla()
        self.ax[2].scatter(
            mlh_model.pos[:, 1],
            mlh_model.pos[:, 0],
            mlh_model.pos[:, 2],
            c="black",
            s=2,
        )
        # add mlh location to the graph
        self.ax[2].scatter(
            mlh["location"][1], mlh["location"][0], mlh["location"][2], c="red", s=15
        )
        self.ax[2].set_title("MLH")
        self.ax[2].set_axis_off()
        self.ax[2].set_aspect("equal")

    def add_text(
        self,
        mlh,
        pos,
        possible_matches,
        graph_ids,
        evidences,
    ):
        if self.text:
            self.text.remove()
        new_text = r"MLH of first LM: "
        mlh_id = mlh["graph_id"].split("_")
        for word in mlh_id:
            new_text += r"$\bf{" + word + "}$ "
        new_text += f"with evidence {np.round(mlh['evidence'], 2)}\n\n"

        # Highlight 2nd MLH if present
        if len(evidences) > 1:
            top_indices = np.flip(np.argsort(evidences))[0:2]
            second_id = graph_ids[top_indices[1]].split("_")
            new_text += "2nd MLH: "
            for word in second_id:
                new_text += r"$\bf{" + word + "}$ "
            new_text += f"with evidence {np.round(evidences[top_indices[1]], 2)}\n\n"

        new_text += r"$\bf{Possible}$ $\bf{matches:}$"
        for gid, ev in zip(graph_ids, evidences):
            if gid in possible_matches:
                new_text += f"\n{gid}: {np.round(ev, 1)}"

        self.text = self.ax[0].text(0, pos + 30, new_text, va="top")

    def setup_camera_ax(self):
        self.ax[0].set_title("Camera image")
        self.ax[0].set_axis_off()
        self.camera_image = None
        self.text = None

    def setup_sensor_ax(self):
        self.ax[1].set_title("Sensor image")
        self.ax[1].set_axis_off()
        self.sensor_image = None

    def setup_mlh_ax(self):
        self.ax[2] = plt.subplot(1, 3, 3, projection="3d")
        self.ax[2].set_title("MLH")
        self.ax[2].set_axis_off()
        self.ax[2].set_aspect("equal")
