# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""A collection of plot utilities not used during normal platform runtime.

Do not read too much into the "_dev" suffix of this file. We needed separate
files to segment what is imported.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Sequence, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.spatial.transform import Rotation
from torch_geometric.data import Data

from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.object_model import GraphObjectModel
from tbp.monty.frameworks.utils.graph_matching_utils import find_step_on_new_object
from tbp.monty.frameworks.utils.plot_utils import (
    add_patch_outline_to_view_finder,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import get_angle
from tbp.monty.frameworks.utils.transform_utils import numpy_to_scipy_quat

if TYPE_CHECKING:
    from numbers import Number


def plot_graph(
    graph: Data | GraphObjectModel,
    show_nodes: bool = True,
    show_edges: bool = False,
    show_trisurf: bool = False,
    show_axticks: bool = False,
    rotation: Number = -80,
    ax_lim: Sequence | None = None,
    ax: Axes3D | None = None,
) -> Figure:
    """Plot a 3D graph of an object model.

    TODO: add color_by option

    Args:
        graph: The graph object that should be visualized.
        show_nodes: Whether to plot the nodes of the graph.
        show_edges: Whether to plot the displacements between nodes as edges.
        show_trisurf: Whether to plot the trisurface plot along the graphs nodes.
        show_axticks: Whether to show axis ticks.
        rotation: Rotation of the 3D plot (moving camera up or down).
        ax_lim: axis limit for x and y axis (i.e. resolution).
        ax: Axes3D instance to plot on. If not supplied, a figure and Axes3D
            instance will be created.

    Returns:
        The figure on which the graph was plotted.

    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="3d")
    else:
        fig = ax.figure

    if show_nodes:
        ax.scatter(graph.pos[:, 1], graph.pos[:, 0], graph.pos[:, 2], c=graph.pos[:, 2])

    if show_edges:
        for i, e in enumerate(graph.edge_index[0]):
            e2 = graph.edge_index[1][i]
            ax.plot(
                [graph.pos[e, 1], graph.pos[e2, 1]],
                [graph.pos[e, 0], graph.pos[e2, 0]],
                [graph.pos[e, 2], graph.pos[e2, 2]],
                color="tab:gray",
            )

    if show_trisurf:
        ax.plot_trisurf(graph.pos[:, 1], graph.pos[:, 0], graph.pos[:, 2], alpha=0.7)

    if not show_axticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel("x", labelpad=-10)
        ax.set_zlabel("z", labelpad=-15)
        ax.set_ylabel("y", labelpad=-15)
    else:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    if ax_lim is not None:
        ax.set_xlim([0, ax_lim])
        ax.set_ylim([ax_lim, 0])
    else:
        ax.set_aspect("equal")
    ax.view_init(rotation, 180)
    fig.tight_layout()
    return fig


def get_model_id(epoch, mode):
    if epoch == 0:
        model_id = "pretrained"
    elif mode == "eval":
        model_id = str(epoch)
    else:
        model_id = str(epoch - 1)
    return model_id


def get_action_name(
    action_stats: list[list[Action | dict[str, Any] | None]],
    step: int,
    is_match_step: bool,
    obs_on_object: bool,
) -> str:
    """Get the name of the action taken at a step.

    When the name is derived from an action, the format is:

        "{action.name} - {param1}:{value1},{param2}:{value2},..."

    e.g.:

        "move_tangentially - distance:13,direction:[1, 2, 3]"

    Args:
        action_stats: Action statistics.
        step: Step number.
        is_match_step: Whether the step is a match step.
        obs_on_object: Whether the observations are on the object.


    Returns:
        Action name or one of the following sentinel values: "updating possible
        matches", "patch not on object", "not moved yet", "None"
    """
    if is_match_step:
        if obs_on_object:
            action_name = "updating possible matches"
        else:
            action_name = "patch not on object"
    elif step == 0:
        action_name = "not moved yet"
    else:
        action = action_stats[step - 1]
        if action[0] is not None:
            a = cast("Action", action[0])
            d = dict(a)
            del d["action"]  # don't duplicate action in "params"
            del d["agent_id"]  # don't duplicate agent_id in "params"
            params = [
                f"{k}:{v.tolist()}" if isinstance(v, np.ndarray) else f"{k}:{v}"
                for k, v in d.items()
            ]
            action_name = f"{a.name} - {','.join(params)}"
        else:
            action_name = "None"
    return action_name


def set_target_text(ax, target, num_objects):
    ax.text(0, 0.3, f"target: {target['object']}")
    target_rotation_euler = target["euler_rotation"]
    if num_objects > 2:
        text_loc = 0.0
    else:
        text_loc = 0.07
    ax.text(
        text_loc,
        0.02,
        f"rotation: {target_rotation_euler}\n"
        f"position: {target['position']}\n"
        f"scale: {target['scale']}",
    )
    if num_objects == 2:
        ax.text(-0.05, 1.6, "rgba")
        ax.text(-0.06, 1.4, "depth")
    ax.axis("off")


def format_ax(ax, all_model_pos, ax_range, rotate, step):
    if ax_range is None:
        ax.set_xlim([min(all_model_pos[1:, 0]), max(all_model_pos[1:, 0])])
        ax.set_ylim([min(all_model_pos[1:, 1]), max(all_model_pos[1:, 1])])
        ax.set_zlim([min(all_model_pos[1:, 2]), max(all_model_pos[1:, 2])])
    else:
        ax.set_xlim(
            [
                np.mean(all_model_pos[1:, 0]) - ax_range,
                np.mean(all_model_pos[1:, 0]) + ax_range,
            ]
        )
        ax.set_ylim(
            [
                np.mean(all_model_pos[1:, 1]) - ax_range,
                np.mean(all_model_pos[1:, 1]) + ax_range,
            ]
        )
        ax.set_zlim(
            [
                np.mean(all_model_pos[1:, 2]) - ax_range,
                np.mean(all_model_pos[1:, 2]) + ax_range,
            ]
        )

    ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
    ax.set_xlabel("x", labelpad=-10)
    ax.set_ylabel("y", labelpad=-10)
    ax.set_zlabel("z", labelpad=-10)
    if rotate:
        ax.view_init(30, 30 + step * 5)
    else:
        ax.view_init(30, 30)


def get_search_positions(start_node, possible_poses, displacement):
    search_positions = []
    for pose in possible_poses:
        ref_frame_rot = Rotation.from_euler("xyz", pose, degrees=True)

        search_pos = start_node + ref_frame_rot.apply(displacement)
        search_positions.append(search_pos)
    return np.array(search_positions)


def get_model_colors(shape, step, is_match_step):
    if step == 0 and is_match_step is False:
        model_colors = np.ones(shape)
    else:
        model_colors = np.zeros(shape)
    return model_colors


def get_match_step(is_match_step, step):
    if not is_match_step and step > 0:
        return step - 1

    return step


def plot_search_displacements(ax, search_positions, start_node):
    # Plot possible next locations
    ax.scatter(
        search_positions[:, 0],
        search_positions[:, 1],
        search_positions[:, 2],
        s=5,
        c="grey",
    )
    # Plot possible pose displacements
    for sp in search_positions:
        ax.plot3D(
            [start_node[0], sp[0]],
            [start_node[1], sp[1]],
            [start_node[2], sp[2]],
            "lightgray",
        )


def plot_previous_path(ax, current_path, step):
    if step > 0:
        for n in range(len(current_path) - 1):
            ax.plot3D(
                [current_path[n][0], current_path[n + 1][0]],
                [current_path[n][1], current_path[n + 1][1]],
                [current_path[n][2], current_path[n + 1][2]],
                c="green",
            )


def show_previous_possible_paths_with_nodes(ax, path_stats, step, object_n):
    if step > 0:
        previous_paths = path_stats[step - 1][object_n]
        for path in previous_paths:
            plot_previous_path(ax, path, step)
            previous_node_id = path[-1]
            ax.scatter(
                previous_node_id[0],
                previous_node_id[1],
                previous_node_id[2],
                s=30,
                c="green",
                vmin=0,
            )


def update_text(text_ax, num_possible_paths, unique_poses):
    num_unique_poses = unique_poses.shape[0]
    if num_unique_poses > 4:
        text_ax.set_text(f"# paths: {num_possible_paths}\n# poses: {num_unique_poses}")
    else:
        text = f"# paths: {num_possible_paths}\n# poses: {num_unique_poses}"
        for pose in unique_poses:
            text = text + "\n" + str(pose)
        text_ax.set_text(text)


def plot_normal(ax, start_loc, norm, norm_len, color):
    ax.plot(
        [
            start_loc[0],
            start_loc[0] + norm[0] * norm_len,
        ],
        [
            start_loc[1],
            start_loc[1] + norm[1] * norm_len,
        ],
        [
            start_loc[2],
            start_loc[2] + norm[2] * norm_len,
        ],
        c=color,
    )


def show_one_step(
    stats,
    lm_models,
    episode,
    step,
    target_rotation,
    object_to_inspect,
    lm_id="LM_0",
    lm_num=0,
    object_name=None,
    show_num_pos=None,
    show_full_path=False,
    color_by_curvature=False,
    show_surface_normals=False,
    norm_len=0.01,
    ax_range=0.05,
):
    """Shows matching procedure for one specific time step.

    Best used in a notebook with `%matplotlib notebook` to rotate and zoom on the 3d
    plot.
    """
    epoch = stats[str(episode)][lm_id]["train_epochs"]
    model_id = get_model_id(epoch, stats[str(episode)][lm_id]["mode"])
    if object_name is None:
        object_name = object_to_inspect
    model_pos = lm_models[model_id][lm_num][object_to_inspect].pos.numpy()
    model_features = lm_models[model_id][lm_num][object_to_inspect].x.numpy()
    model_normals = lm_models[model_id][lm_num][object_to_inspect].norm.numpy()
    model_f_mapping = lm_models[model_id][lm_num][object_to_inspect].feature_mapping
    first_input_channel = list(model_f_mapping.keys())[0]

    fig = plt.figure(figsize=(7, 7))
    fig.tight_layout()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.axis("off")

    if color_by_curvature:
        curvature_ids = model_f_mapping[first_input_channel]["principal_curvatures"]
        model_curvatures = model_features[:, curvature_ids[0] : curvature_ids[1]]
        ax.scatter(
            model_pos[:, 0],
            model_pos[:, 1],
            model_pos[:, 2],
            alpha=0.7,
            s=5,
            vmin=-500,
            vmax=500,
            cmap="seismic",
            c=model_curvatures[:, 0] * model_curvatures[:, 1],
        )
    else:
        ax.scatter(
            model_pos[:, 0],
            model_pos[:, 1],
            model_pos[:, 2],
            alpha=0.7,
            s=5,
            c="grey",
        )

    displacement = stats[str(episode)][lm_id]["displacement"][step + 1]
    qcurve = stats[str(episode)][lm_id]["principal_curvatures"][step + 1]
    print(f"query curvature: {qcurve}")
    observed_surface_normal = stats[str(episode)][lm_id]["pose_vectors"][step + 1][:3]

    num_possible_paths = len(
        stats[str(episode)][lm_id]["possible_poses"][step][object_to_inspect]
    )
    for path in range(num_possible_paths):
        current_path = stats[str(episode)][lm_id]["possible_paths"][step][
            object_to_inspect
        ][path]
        possible_poses = stats[str(episode)][lm_id]["possible_poses"][step][
            object_to_inspect
        ][path]
        start_node = current_path[-1]
        if show_full_path:
            plot_previous_path(ax, current_path, step)
        ax.scatter(
            start_node[0],
            start_node[1],
            start_node[2],
            s=40,
            c="green",
            vmin=0,
        )

        search_positions = []

        for pose in possible_poses[:show_num_pos]:
            ref_frame_rot = Rotation.from_euler("xyz", pose, degrees=True)

            search_pos = start_node + ref_frame_rot.apply(displacement)

            search_positions.append(search_pos)

            node_distances = np.linalg.norm(
                model_pos - search_pos,
                axis=1,
            )
            on_same_surface_side = False
            while not on_same_surface_side:
                closest_node_id = node_distances.argmin()
                graph_surface_normal = model_normals[closest_node_id].copy()
                graph_surface_normal = ref_frame_rot.apply(graph_surface_normal)
                angle = get_angle(graph_surface_normal, observed_surface_normal)
                angle = (angle - np.pi) % np.pi
                on_same_surface_side = angle < np.pi / 2
                if not on_same_surface_side:
                    wrong_side_node = model_pos[closest_node_id]
                    ax.scatter(
                        wrong_side_node[0],
                        wrong_side_node[1],
                        wrong_side_node[2],
                        c="red",
                        s=30,
                    )
                    # print(f"nearest point not on right surface side for pose {pose}")
                    node_distances[closest_node_id] = 1000000000
                    if np.min(node_distances) == 1000000000:
                        print("no point on right surface side found")
                        break
            closest_node_position = model_pos[closest_node_id]

            if pose == target_rotation:
                print(
                    f"---\npose {pose} with closest node {closest_node_id} \n"
                    f"at position {closest_node_position} \n"
                    f"with distance {node_distances[closest_node_id]} \n"
                    f"has feature {model_features[closest_node_id]}"
                )
                color = "limegreen"
                size = 40
            else:
                color = "pink"
                size = 10
            ax.scatter(
                closest_node_position[0],
                closest_node_position[1],
                closest_node_position[2],
                c=color,
                s=size,
            )
            if show_surface_normals:
                norm = model_normals[closest_node_id]
                # print("norm at closest node (black): " + str(norm))
                # print(
                #     "rotated norm (pink): "
                #     + str(graph_surface_normal)
                #     + " for pose "
                #     + str(pose)
                # )
                # print("observed norm: " + str(observed_surface_normal))
                # print("angle between rotated and observed: " + str(angle))
                plot_normal(ax, closest_node_position, norm, norm_len, "black")
                plot_normal(
                    ax, closest_node_position, graph_surface_normal, norm_len, "pink"
                )

        search_positions = np.array(search_positions)
        plot_search_displacements(ax, search_positions, start_node)

        ax.set_title(f"Step {step} - {object_name}")
        format_ax(ax, model_pos, ax_range, rotate=False, step=step)


def show_initial_hypotheses(
    detailed_stats,
    episode,
    obj,
    axis=0,
    lm="LM_0",
    rotation=(80, 0),
    save_fig=False,
    save_path="./",
):
    """Plot the initial rotation hypotheses for an object.

    Args:
        detailed_stats: run stats loaded from json log file.
        episode: episode from which the initial hypotheses shouls be shown.
        obj: ovject model to visualize.
        axis: rotation along which axis should be visualized. Defaults to 0.
        lm: lm ID from which to show the model and hypotheses. Defaults to "LM_0".
        rotation: initial rotation of the 3D plot. Defaults to (80, 0).
        save_fig: whether to save the plot at save_path. Defaults to False.
        save_path: where to save the plot. Defaults to "./".
    """
    fig = plt.figure()
    lm_stats = detailed_stats[str(episode)][lm]
    locs = np.array(lm_stats["possible_locations"][0][obj])
    colors = np.array(
        [
            Rotation.from_matrix(pose).as_euler("xyz", degrees=True)[axis]
            for pose in lm_stats["possible_rotations"][0][obj]
        ]
    )
    ax = plt.subplot(1, 1, 1, projection="3d")
    s = ax.scatter(
        locs[:, 0],
        locs[:, 1],
        locs[:, 2],
        c=colors,
        cmap="hsv",
        vmin=-180,
        vmax=180,
    )
    fig.colorbar(s)
    ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
    ax.set_aspect("equal")
    ax.view_init(rotation[0], rotation[1])
    possible_ax = ["x", "y", "z"]
    plt.title(f"Possible Rotations along {possible_ax[axis]} axis")
    if save_fig:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print("figure saved at " + save_path)
        plt.savefig(
            save_path + f"initialH_{episode}_{obj}_{possible_ax[axis]}.png",
            bbox_inches="tight",
            dpi=300,
        )
    else:
        plt.show()


def plot_evidence_at_step(
    detailed_stats,
    lm_models,
    episode,
    step,
    objects,
    lm="LM_0",
    sm="SM_0",
    view_finder="SM_1",
    input_feature_channel="patch",
    current_evidence_update_threshold=-1,
    is_surface_sensor=False,
    is_2d_image_movement=False,
    save_fig=False,
    save_path="./",
):
    """Plot evidence for all hypotheses at one step during matching with evidence LM.

    Args:
        detailed_stats: run stats loaded from json log file.
        lm_models: object models (graphs) from this experiment.
        episode: episode to visualize.
        step: step to visualize.
        objects: objects to visualize (has to be len=4).
        lm: learning module to visualize. Defaults to "LM_0".
        sm: sensor module to visualize. Defaults to "SM_0".
        view_finder: view finder. Defaults to "SM_1".
        input_feature_channel: input feature channel. Defaults to "patch".
        current_evidence_update_threshold: threshold at which alpha value should be
            0.5 instead of 1. Defaults to -1.
        is_surface_sensor: ?. Defaults to False.
        is_2d_image_movement: ?. Defaults to False.
        save_fig: Whether to save the plot at save_path. Defaults to False.
        save_path: location to save the plot at. Defaults to "./".
    """
    lm_stats = detailed_stats[str(episode)][lm]
    sm_stats = detailed_stats[str(episode)][sm]
    view_finder_obs = detailed_stats[str(episode)][view_finder]["raw_observations"]
    pose_colors = ["blue", "red", "orange"]

    plt.figure(figsize=(20, 10))
    qf = np.array(lm_stats[input_feature_channel]["pose_vectors"][step]).reshape(
        (
            3,
            3,
        )
    )
    ax = plt.subplot(2, 4, 1, projection="3d")
    loc = np.array([0, 0, 0])
    ax.scatter(0, 0, 0, c="red", s=10)
    if step > 0:
        for disp in lm_stats["displacements"][input_feature_channel]["displacement"][
            : step + 1
        ]:
            next_loc = loc + disp
            ax.plot(
                [loc[0], next_loc[0]],
                [loc[1], next_loc[1]],
                [loc[2], next_loc[2]],
                c="black",
            )
            loc = next_loc
    for n, p in enumerate(np.array(qf)):
        ax.plot(
            [loc[0], loc[0] + p[0] * 0.01],
            [loc[1], loc[1] + p[1] * 0.01],
            [loc[2], loc[2] + p[2] * 0.01],
            c=pose_colors[n],
        )
    ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
    ax.set_aspect("equal")
    plt.title("displacments \n+ pose feature")
    plt.suptitle(f"Step {step}", fontsize=20)
    if is_surface_sensor:
        sm_step = step * 4 + 3  # get observation after 3 corrective movements
    else:
        sm_step = step
    if "rgba" in view_finder_obs[sm_step].keys():
        obs_key = "rgba"
    else:
        obs_key = "depth"
    ax = plt.subplot(2, 4, 2)
    view_finder_image = view_finder_obs[sm_step][obs_key]
    patch_image = sm_stats["raw_observations"][sm_step][obs_key]
    if is_2d_image_movement:
        center_pixel_id = np.array(sm_stats["raw_observations"][step]["pixel_loc"])
        patch_size = np.array(patch_image).shape[0]
        view_finder_image = add_patch_outline_to_view_finder(
            view_finder_image, center_pixel_id, patch_size
        )
    plt.imshow(view_finder_image)
    plt.title("view finder")
    plt.axis("off")
    ax = plt.subplot(2, 4, 3)
    plt.imshow(patch_image)
    plt.title("sensor patch")
    plt.axis("off")
    ax = plt.subplot(2, 4, 4)
    mlh = lm_stats["current_mlh"][step]
    if mlh is None:
        mlh = {"graph_id": None, "rotation": [0, 0, 0], "evidence": 0}
    mlhr = np.round(lm_stats["target"]["euler_rotation"], 1)
    text = (
        f"target: \n{lm_stats['target']['object']}"
        f" with rotation {mlhr}\n"
        "\ncurrent most likely hypothesis:\n"
        f"{mlh['graph_id']} with rotation {np.round(mlh['rotation'], 1)}"
        f"\nevidence: {np.round(mlh['evidence'], 2)}"
    )
    ax.text(0, 0.5, text)
    plt.axis("off")

    max_evidence = -np.inf
    min_evidence = np.inf
    for obj in objects:
        evidences = np.array(lm_stats["evidences"][step][obj])
        if np.max(evidences) > max_evidence:
            max_evidence = np.max(evidences)
        if np.min(evidences) < min_evidence:
            min_evidence = np.min(evidences)

    for n, obj in enumerate(objects):
        #     continue
        locs = np.array(lm_stats["possible_locations"][step][obj])
        if "pretrained" in lm_models.keys():
            model_pos = lm_models["pretrained"][0][obj][
                input_feature_channel
            ].pos  # TODO: test
        elif str(episode - 1) in lm_models.keys():
            model_pos = lm_models[str(episode - 1)][lm][obj].pos
        else:
            last_stored_models = np.max(np.array(list(lm_models.keys()), dtype=int))
            model_pos = lm_models[str(last_stored_models)][lm][obj][
                input_feature_channel
            ].pos
        evidences = np.array(lm_stats["evidences"][step][obj])
        colors = evidences
        sizes = np.array(lm_stats["evidences"][step][obj]) * 10
        sizes[sizes <= 0] = 0.1
        ids_not_updated = evidences < current_evidence_update_threshold
        alphas = np.ones(colors.shape)
        alphas[ids_not_updated] = 0.5

        ax2 = plt.subplot(2, 4, 5 + n, projection="3d")
        _ = ax2.scatter(
            locs[:, 0],
            locs[:, 1],
            locs[:, 2],
            c=colors,
            cmap="seismic",
            alpha=alphas,
            vmin=min_evidence,
            vmax=max_evidence,
            s=sizes,
        )
        _ = ax2.scatter(
            model_pos[:, 0],
            model_pos[:, 1],
            model_pos[:, 2],
            c="grey",
            s=2,
            alpha=0.5,
        )
        ax2.set_aspect("equal")
        # fig.colorbar(s)
        ax2.set_xticks([]), ax2.set_yticks([]), ax2.set_zticks([])

    if save_fig:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print("figure saved at " + save_path)
        plt.savefig(
            save_path + f"{episode}_{step}.png",
            bbox_inches="tight",
            dpi=300,
        )
    else:
        plt.show()


class PolicyPlot:
    """Used to plot the path taken by the agent over the ground-truth object.

    This includes the estimated location (based on the depth sensor) on the object.
    For the surface ("finger") policy, the agent will typically traverse the object; for
    the distant ("eye") policy, the agent will perform saccade-like movements over an
    object following a random walk. In addition, both agents now have the ability to
    perform hypothesis-testing jumps onto another part of the object.
    """

    def __init__(
        self,
        detailed_stats,
        lm_models,
        episode,
        object_id,
        agent_type="surface",
        jumps_used=False,
        extra_vis="lm_processed",
        lm_index=0,
    ):
        """Initialize the PolicyPlot class.

        Args:
            detailed_stats: Detailed stats for the experiment.
            lm_models: The learned models from pre-training; will be used to
                visualize the object in the environment.
            episode: The episode number to visualize.
            object_id: pass in the ground-truth object ID for the episode.
            agent_type: Specify whether the agent type is the "distant" (i.e.
                eye-like) or "surface" (i.e. finger-like) agent; affects what
                information we expect to have available for plotting.
            jumps_used: Whether the hypothesis-testing "jumping" action policy was used,
                and therefore whether to attempt visualizing relevant policy steps.
            extra_vis: Can be set to either "lm_processed", "sensor_pose", or None.
            If lm_processed, performs additional visualization of the sensations
            associated with passing information to the learning-module
            If sensor_pose, visualizes the location and orientation of the sensor during
                the episode.
            lm_index: The index of the learning module we wish to visualize the
                relevant policy for; NOTE no testing has been done for multi-LM setups
                or heterarchy.
        """
        self.detailed_stats = detailed_stats
        self.lm_models = lm_models
        self.episode = episode
        self.object_id = object_id
        self.agent_type = agent_type
        self.jumps_used = jumps_used
        self.extra_vis = extra_vis
        self.lm_index = lm_index

    def plot_core_object(self):
        """Plot the core object model.

        Note that all coordinates used for plotting are relative to the world
        coordinates, hence e.g. surface normals do not need to be rotated by the
        object's orientation in the environment; the only rotation that needs to be done
        therefore is to get the learned object points (in their arbitrary, internal
        reference frame) to align with the actual rotation of the object in the
        environment
        """
        self.fig = plt.figure(figsize=(5, 5))
        self.ax = plt.subplot(1, 1, 1, projection="3d")

        # Use point-cloud model of ground-truth object that is in the evironment
        # This is based on the *LM's model*, but always getting the ground-truth object,
        # i.e. regardless of whether the LM is successfully recognizing the object or
        # not
        # Thus we can see if e.g. there is a difference in exploration depending on how
        # well known areas on the model are
        learned_model_cloud = self.lm_models["pretrained"][self.lm_index][
            self.object_id
        ].pos

        converted_quat = numpy_to_scipy_quat(
            self.detailed_stats[str(self.episode)]["target"][
                "primary_target_rotation_quat"
            ]
        )
        object_rot = Rotation.from_quat(converted_quat)

        # Update orientation and position of the learned model
        # to be consistent with [lm]["locations"] (which are in body-centric
        # coordinates)
        learned_model_cloud = (
            object_rot.apply(learned_model_cloud)
            + self.detailed_stats[str(self.episode)]["target"][
                "primary_target_position"
            ]
        )

        self.ax.scatter(
            learned_model_cloud[:, 0],
            learned_model_cloud[:, 1],
            learned_model_cloud[:, 2],
            c=learned_model_cloud[:, 2],
            alpha=0.3,
        )

    def derive_policy_details(self):
        """Derive any data-structures of interest for later plotting.

        Steps associated with particular types of movement is an example of a
        data-structure of interest.

        In general, we focus on visualizing steps associated with exploring the object
        (e.g. tangential steps for the surface-agent), ignoring "corrective" steps, such
        as moving back onto the object, or re-orienting the surface agent to be
        normal to the surface
        """
        # Get the list of all observation locations, i.e. all observations experienced
        # by the sensory module; this will be a superset of locations experienced
        # by the learning module
        sm_locs = []
        for current_obs in self.detailed_stats[str(self.episode)][
            "SM_" + str(self.lm_index)
        ]["processed_observations"]:
            sm_locs.append(current_obs.location)
        sm_locs = np.array(sm_locs)

        # 1D "mask" that is of the length of the full number of observations, and which
        # is True where the observation was passed to the learning module
        self.lm_steps_mask = np.array(
            self.detailed_stats[str(self.episode)]["LM_" + str(self.lm_index)][
                "lm_processed_steps"
            ]
        )

        # Get the a boolean mask of the observations that were associated with
        # tangential movements (i.e. meaningfully exploratory rather than corrective
        # movements)
        self.tangential_steps_mask = []
        if self.agent_type == "surface":
            for current_action in self.detailed_stats[str(self.episode)][
                "motor_system"
            ]["action_sequence"]:
                if current_action[0] is not None:
                    self.tangential_steps_mask.append(
                        "move_tangentially" in current_action[0]
                    )
                else:
                    # First movement is associated with [None, None]
                    self.tangential_steps_mask.append(False)
            self.tangential_steps_mask = np.array(self.tangential_steps_mask)
        elif self.agent_type == "distant":
            # For distant-agent, we count any steps that were on the object as
            # "tangential"
            for current_obs in self.detailed_stats[str(self.episode)][
                "SM_" + str(self.lm_index)
            ]["processed_observations"]:
                self.tangential_steps_mask.append(current_obs["features"]["on_object"])

        # Get the *locations* associated with tangential movements
        self.tangential_locs = sm_locs[np.where(self.tangential_steps_mask)]

        # Derive arrays for coloring the markers of tangential moves
        # based on their context in the action policy; currently only significant
        # for the surface policy
        if self.agent_type == "surface":
            self.pc_headings = np.array(
                self.detailed_stats[str(self.episode)]["motor_system"][
                    "action_details"
                ]["pc_heading"]
            )
            self.new_headings = np.array(
                self.detailed_stats[str(self.episode)]["motor_system"][
                    "action_details"
                ]["avoidance_heading"]
            )
            assert len(self.tangential_locs) == len(self.new_headings), (
                "Mismatch in number of heading-markers"
            )

        else:
            self.pc_headings = np.zeros(len(self.tangential_locs))
            self.new_headings = np.zeros(len(self.tangential_locs))

    def plot_movement_step(self, step_iter):
        """Plot the action policy behavior associated with a particular step."""
        prev_loc = self.tangential_locs[step_iter]
        new_loc = self.tangential_locs[step_iter + 1]

        # Recover the episode step assocaited with the current tangential step
        episode_step = np.where(self.tangential_steps_mask)[0][step_iter]

        # Check if there was a successful jump
        # Logic is - first check if jumps used at all, then if a jump was attempted,
        # and then whether the jump was succsseful
        if self.jumps_used:
            jump_attempted_bool = (
                episode_step
                in self.detailed_stats[str(self.episode)]["motor_system"][
                    "action_details"
                ]["episode_step_for_jump"]
            )
            if jump_attempted_bool:
                idx_jump = np.where(
                    np.array(
                        self.detailed_stats[str(self.episode)]["motor_system"][
                            "action_details"
                        ]["episode_step_for_jump"]
                    )
                    == episode_step
                )[0][0]
                jump_successful = self.detailed_stats[str(self.episode)][
                    "motor_system"
                ]["action_details"]["jump_successful"][idx_jump]
            else:
                jump_successful = False
        else:
            jump_successful = False

        if jump_successful:
            color = "purple"
            alpha = 0.75
        elif self.new_headings[step_iter + 1]:
            # Note this takes presedence over pc-heading
            color = "lawngreen"
            alpha = 0.5
        elif self.pc_headings[step_iter + 1] == "min":
            color = "white"
            alpha = 0.5
        elif self.pc_headings[step_iter + 1] == "max":
            color = "black"
            alpha = 0.5
        else:
            color = "dodgerblue"
            alpha = 0.5

        self.ax.plot(
            [prev_loc[0], new_loc[0]],
            [prev_loc[1], new_loc[1]],
            [prev_loc[2], new_loc[2]],
            c=color,
            alpha=alpha,
            linewidth=4,
            marker="^",
            markersize=6,
        )

        # If on a step associated with a failed jump, also visualize this
        if self.jumps_used:
            if (
                episode_step
                in self.detailed_stats[str(self.episode)]["motor_system"][
                    "action_details"
                ]["episode_step_for_jump"]
            ):
                self.check_failed_jump(episode_step)

        # Additional, optional visualizations
        if self.extra_vis == "lm_processed":
            self.add_lm_processing(step_iter)

        elif self.extra_vis == "sensor_pose":
            self.add_sensor_scatter(step_iter + 1)

    def check_failed_jump(self, step_iter):
        """Visualize where in space an agent attempted a failed jump to.

        Only unsuccessful, hypothesis-testing jumps are visualized.

        Note this uses a more verbose method of defining the sensor pose, using
        information saved about both the agent and sensor pose, in contrast to
        add_sensor_scatter; this was implemented so that if desired, one could
        separately visualize the pose of the agent, and the sensor
        relative to it, although currently this is not used

        TODO fix this to always work with the surface agent - currently,
        if a failed jump is not associated with a tangential step, then we won't
        visualize it
        """
        idx_jump = np.where(
            np.array(
                self.detailed_stats[str(self.episode)]["motor_system"][
                    "action_details"
                ]["episode_step_for_jump"]
            )
            == step_iter
        )[0][0]

        # Jump was a failure
        if not self.detailed_stats[str(self.episode)]["motor_system"]["action_details"][
            "jump_successful"
        ][idx_jump]:
            # === DETERMINE AGENT POSE ===
            # The location and rotation of the agent (temporarily) before it jumped back
            temp_agent_loc = self.detailed_stats[str(self.episode)]["motor_system"][
                "action_details"
            ]["post_jump_pose"][idx_jump][AgentID("agent_id_0")]["position"]

            temp_agent_rot = Rotation.from_quat(
                numpy_to_scipy_quat(
                    self.detailed_stats[str(self.episode)]["motor_system"][
                        "action_details"
                    ]["post_jump_pose"][idx_jump][AgentID("agent_id_0")]["rotation"]
                )
            )

            # === PLOT SENSOR POSE ===
            # Sensor(s)'s effective pose in the environment accounts for the position of
            # the agent, as its coordinates are relative to it, not the environment

            # Get the relevant sensors; excludes viewfinder
            sensors_to_plot = [
                x
                for x in self.detailed_stats[str(self.episode)]["motor_system"][
                    "action_details"
                ]["post_jump_pose"][idx_jump][AgentID("agent_id_0")]["sensors"]
                if "patch" in x and ".depth" in x
            ]

            for sensor_key in sensors_to_plot:
                temp_sensor_loc = np.array(temp_agent_loc) + np.array(
                    self.detailed_stats[str(self.episode)]["motor_system"][
                        "action_details"
                    ]["post_jump_pose"][idx_jump][AgentID("agent_id_0")]["sensors"][
                        sensor_key
                    ]["position"]
                )

                partial_sensor_rot = Rotation.from_quat(
                    numpy_to_scipy_quat(
                        self.detailed_stats[str(self.episode)]["motor_system"][
                            "action_details"
                        ]["post_jump_pose"][idx_jump][AgentID("agent_id_0")]["sensors"][
                            sensor_key
                        ]["rotation"]
                    )
                )
                temp_sensor_rot = (
                    temp_agent_rot * partial_sensor_rot
                )  # Compose the rotations

                # As the sensor faces "forward" along the negative z-axis, we use this
                # vector to visualize its orientation
                sensor_direction = temp_sensor_rot.apply(np.array([0, 0, -1]))

                scaling = 0.02
                self.ax.plot(
                    [
                        temp_sensor_loc[0],
                        temp_sensor_loc[0] + sensor_direction[0] * scaling,
                    ],
                    [
                        temp_sensor_loc[1],
                        temp_sensor_loc[1] + sensor_direction[1] * scaling,
                    ],
                    [
                        temp_sensor_loc[2],
                        temp_sensor_loc[2] + sensor_direction[2] * scaling,
                    ],
                    linewidth=5,
                    alpha=0.5,
                    c="grey",
                )
                self.ax.scatter(
                    [temp_sensor_loc[0]],
                    [temp_sensor_loc[1]],
                    [temp_sensor_loc[2]],
                    s=70,
                    alpha=0.9,
                    c="black",
                )

    def add_lm_processing(self, step_iter):
        """Visualize the surface normal associated with an LM-processed step."""
        detailed_features = self.detailed_stats[str(self.episode)]["SM_0"][
            "processed_observations"
        ]

        current_loc = self.tangential_locs[step_iter]

        # Compare indices associated with tangential movements and
        # LM processing; when a tangential movement was also
        # associated with sending data to LM, add a surface normal
        if (
            np.where(self.tangential_steps_mask)[0][step_iter]
            in np.where(self.lm_steps_mask)[0]
        ):
            matching_index = np.where(self.tangential_steps_mask)[0][step_iter]

            pose_colors = ["lightblue", "lightcoral", "bisque"]
            qf = np.array(
                detailed_features[matching_index]["features"]["pose_vectors"]
            ).reshape(
                (
                    3,
                    3,
                )
            )

            for n, p in enumerate(np.array(qf)):
                self.ax.plot(
                    [current_loc[0], current_loc[0] + p[0] * 0.01],
                    [current_loc[1], current_loc[1] + p[1] * 0.01],
                    [current_loc[2], current_loc[2] + p[2] * 0.01],
                    c=pose_colors[n],
                )

    def add_sensor_scatter(self, step_iter):
        """Utility to visualize an agent/sensor(s)'s location and orientation."""
        sensors_to_plot = [
            x for x in self.detailed_stats[str(self.episode)].keys() if "SM_" in x
        ]

        for sensor_key in sensors_to_plot:
            sensor_loc = self.detailed_stats[str(self.episode)][sensor_key][
                "sm_properties"
            ][np.where(self.tangential_steps_mask)[0][step_iter]]["sm_location"]
            sensor_rot = Rotation.from_quat(
                numpy_to_scipy_quat(
                    self.detailed_stats[str(self.episode)][sensor_key]["sm_properties"][
                        np.where(self.tangential_steps_mask)[0][step_iter]
                    ]["sm_rotation"]
                )
            )

            # As the agent faces "forward" along the negative z-axis, we use this vector
            # to visualize its orientation
            sensor_direction = sensor_rot.apply(np.array([0, 0, -1]))

            scaling = 0.02
            self.ax.plot(
                [sensor_loc[0], sensor_loc[0] + sensor_direction[0] * scaling],
                [sensor_loc[1], sensor_loc[1] + sensor_direction[1] * scaling],
                [sensor_loc[2], sensor_loc[2] + sensor_direction[2] * scaling],
                linewidth=5,
                alpha=0.5,
                c="dodgerblue",
            )
            self.ax.scatter(
                [sensor_loc[0]],
                [sensor_loc[1]],
                [sensor_loc[2]],
                s=70,
                alpha=0.8,
                c="red",
            )

    def plot_up_to_step(self, total_steps=None):
        """Plot the action policy with a static 3D graph up to the step of interest.

        If total_steps is not specified, then plot the full policy (i.e. entire episode)
        """
        self.plot_core_object()
        self.derive_policy_details()

        if total_steps is None:
            total_steps = len(self.tangential_locs) - 1

        for current_step in range(total_steps):
            self.plot_movement_step(current_step)

    def plot_animation(self, _zoom=1.0, view=None):
        """Plot an animation of the episode's full action policy."""
        self.plot_core_object()
        self.derive_policy_details()

        self.ax.set_aspect("equal")
        # self.ax.set_box_aspect((6, 6, 6), zoom=zoom)
        if view is not None:
            self.ax.view_init(view[0], view[1])

        # Create the animation
        ani = animation.FuncAnimation(
            self.fig,
            self.plot_movement_step,
            frames=range(len(self.tangential_locs) - 1),
            repeat=False,
            interval=200,
        )
        ani.save(
            "policy_gif_" + self.object_id + "_episode_" + str(self.episode) + ".gif",
            writer="imagemagick",
            dpi=300,
        )

    def visualize_plot(self, save_path=None):
        self.ax.set_aspect("equal")

        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            print("figure saved at " + save_path)
            plt.savefig(
                save_path + f"{self.episode}.png",
                bbox_inches="tight",
                dpi=300,
            )
        else:
            plt.show()


def plot_learned_graph(
    detailed_stats,
    lm_models,
    episode,
    object_id,
    view=None,
    noise_amount=0.0,
    lm_index=0,
    save_fig=False,
    save_path="./",
):
    """Plot the graph learned for a particular object.

    This plot does not include additional visualizations of policy movements etc.

    It differs from plot_graph in that the focus is on plotting a graph stored in an
    LMs memory, where this is corrected to have the rotation and position in the
    environment as was experienced during an episode.

    Futhermore, there is the option to add noise, such that it is easy to visualize e.g.
    the effect of noise in the location feature.

    Args:
        detailed_stats: ?
        lm_models: ?
        episode: ?
        object_id: ?
        view: the elevation and azimuth to initialize the view at. Defaults to None.
        noise_amount: ?. Defaults to 0.0.
        lm_index: ?. Defaults to 0.
        save_fig: ?. Defaults to False.
        save_path: ?. Defaults to "./".
    """
    # Use point-cloud model of ground-truth object that is in the evironment
    # This is based on the *LM's model*, but always getting the ground-truth object,
    learned_model_cloud = lm_models["pretrained"][lm_index][object_id].pos

    converted_quat = numpy_to_scipy_quat(
        detailed_stats[str(episode)]["target"]["primary_target_rotation_quat"]
    )
    object_rot = Rotation.from_quat(converted_quat)

    # Update orientation and position of the learned model to be in environmental
    # coordinates
    learned_model_cloud = (
        object_rot.apply(learned_model_cloud)
        + detailed_stats[str(episode)]["target"]["primary_target_position"]
    )

    # Add optional noise; can be used to visualize e.g. how significant noise
    # in the sensory information might be
    noise_to_add = np.random.normal(0, noise_amount, size=np.shape(learned_model_cloud))
    learned_model_cloud = learned_model_cloud + noise_to_add

    plt.figure(figsize=(5, 5))
    ax = plt.subplot(1, 1, 1, projection="3d")

    # Plot the learned graph of the object mapped on to where it actually is
    # in the environment
    ax.scatter(
        learned_model_cloud[:, 0],
        learned_model_cloud[:, 1],
        learned_model_cloud[:, 2],
        c=learned_model_cloud[:, 2],
        alpha=0.3,
    )

    ax.set_aspect("equal")
    if view is not None:
        ax.view_init(view[0], view[1])

    if save_fig:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print("figure saved at " + save_path)
        plt.savefig(
            save_path + f"{episode}.png",
            bbox_inches="tight",
            dpi=300,
        )
    else:
        plt.show()


def plot_graph_mismatch(
    second_mlh,
    top_mlh_graph,
    second_mlh_graph,
    displaced_point,
    current_surface_normal,
    save_path,
    plot_name,
    gt_graph=None,
    save_fig=True,
):
    """Plot the mismatch between the most likely and second most likely hypotheses.

    The hypotheses are expected to come from an evidence LM.

    The LM makes use of this mismatch to propose locations to move to and thereby
    test
    """
    # TODO M integrate the below code (moved from evidence-based LM) into this function
    # if plot_graph_bool:
    #     # TODO clean this up to save necessary info to detailed stats,
    #     # and visualize post-hoc
    #     # NB should only generally do this when debugging/visualizing small datasets
    #     # otherwise will end up saving many figures to file
    #     displaced_point = top_mlh_graph[target_loc_id]

    #     save_path = os.path.expanduser(
    #         "~/tbp/results/monty/projects/evidence_eval_runs"
    #         "/logs/hypothesis_driven_pol/episode_" + str(episode_num) + "/"
    #     )

    #     cleaned_quat = str(self.primary_target_rotation_quat).replace(
    #         " ", ","
    #     )
    #     cleaned_quat = cleaned_quat.replace("[", "")
    #     cleaned_quat = cleaned_quat.replace("]", "")
    #     cleaned_quat = cleaned_quat.replace(".", "_")
    #     plot_name = (
    #         "targetID_"
    #         + self.primary_target
    #         + "__mlh_"
    #         + top_mlh["graph_id"]
    #         + "__targetRot_"
    #         + cleaned_quat
    #         + "__lmStep_"
    #         + str(np.sum(self.buffer.stats["lm_processed_steps"]))
    #         + "__epiStep_"
    #         + str(len(self.buffer.stats["lm_processed_steps"]))
    #         + ".png"
    #     )

    #     if gt_graph is not None:
    #         # Perform the same transformation to the ground-truth graph and with
    #         # ground truth pose; used for visualization purposes to see how off the
    #         # internal estimates are
    #         corrected_current_loc = current_env_position - np.array(
    #             [0, 1.5, 0]
    #         )  # TODO
    #         # remove hard-coding of initial object position; note don't need to
    #         # rotate because already fully in environmental-coordinates
    #         gt_graph = gt_rotation.apply(gt_graph) - corrected_current_loc
    #         # Convert from environmental coordinates to the learned coordinate of
    #         # 2nd object
    #         gt_graph = (
    #             second_mlh["rotation"].apply(gt_graph)
    #             + second_mlh["location"]
    #         )

    #     plot_graph_mismatch(
    #         second_mlh,
    #         top_mlh_graph,
    #         second_mlh_graph,
    #         displaced_point,
    #         current_surface_normal,
    #         save_path,
    #         plot_name,
    #         gt_graph=gt_graph,
    #     )

    # == Visualize results ==
    plt.figure(figsize=(5, 5))
    ax = plt.subplot(1, 1, 1, projection="3d")

    # NB how we aim to plot both reference objects in the reference frame of the first
    # object, by correcting the second one
    # Further note the order of indexing the xyz accounts for the y axis being
    # the vertical axis in Habitat; this is also done for e.g. the surface normal
    ax.scatter(
        top_mlh_graph[:, 0],
        top_mlh_graph[:, 2],
        top_mlh_graph[:, 1],
        c=top_mlh_graph[:, 2],
        alpha=0.3,
        cmap="viridis",
    )

    ax.scatter(
        second_mlh_graph[:, 0],
        second_mlh_graph[:, 2],
        second_mlh_graph[:, 1],
        c=(second_mlh_graph[:, 2] * (-1)),
        alpha=0.3,
        cmap="viridis",
    )

    # Optionally plot the mismatch between the estimated pose of the MLH object, and
    # the ground-truth object with its ground-truth pose
    # Helps indicate how often the issue with the jump is that the pose is too
    # wrong
    if gt_graph is not None:
        ax.scatter(
            gt_graph[:, 0],
            gt_graph[:, 2],
            gt_graph[:, 1],
            c=(gt_graph[:, 2] * (-1)),
            alpha=0.3,
            cmap="plasma",
        )

    ax.scatter(
        displaced_point[0],
        displaced_point[2],
        displaced_point[1],
        c="red",
        alpha=0.9,
        s=400,
    )

    base_point = second_mlh["location"]

    ax.scatter(
        base_point[0],
        base_point[2],
        base_point[1],
        c="magenta",
        alpha=0.9,
        s=400,
    )

    # Note that surface normal is in global environmental coordinates, not
    # the environmental coordinates in which the second object was learned,
    # therefore transform by inverse
    second_surface_normal = second_mlh["rotation"].inv().apply(current_surface_normal)
    ax.plot(
        [base_point[0], base_point[0] + second_surface_normal[0] * 0.02],
        [base_point[2], base_point[2] + second_surface_normal[2] * 0.02],
        [base_point[1], base_point[1] + second_surface_normal[1] * 0.02],
        c="pink",
        alpha=0.9,
        linewidth=5,
        label="top_mlh_surface_normal",
    )

    view = [45, -80]
    ax.set_aspect("equal")
    if view is not None:
        ax.view_init(view[0], view[1])

    if save_fig:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(
            fname=save_path + plot_name,
            dpi=300,
        )
    else:
        plt.show()

    plt.clf()


def plot_hotspots(
    detailed_stats,
    lm_models,
    object_id,
    episode_range=1,
    view=None,
    lm_index=0,
    save_fig=False,
    save_path="./",
):
    """Plot the locations frequently visited by the hypothesis-testing jump policy.

    Note the location will not necessarily correspond to where
    the hypothesis-testing policy *thought* it would land; in particular if its
    estimate of the object pose was poor, then it may land somewhere other
    than where it intended

    This function collects and includes data from across multiple episodes; as such
    this visualization should be used when the experiment has involved multiple
    rotations or episodes of only a single object

    Args:
        detailed_stats: ?
        lm_models: ?
        object_id: ?
        episode_range: total number of episodes for a particular object ID that
            should be considered when visualizing hotspots. Defaults to 1.
        view: ?. Defaults to None.
        lm_index: ?. Defaults to 0.
        save_fig: ?. Defaults to False.
        save_path: ?. Defaults to "./".
    """
    # Visualize the object itself
    learned_model_cloud = lm_models["pretrained"][lm_index][object_id].pos
    plt.figure(figsize=(5, 5))
    ax = plt.subplot(1, 1, 1, projection="3d")

    ax.scatter(
        learned_model_cloud[:, 0],
        learned_model_cloud[:, 2],  # Index like this so that z (forward-backward in
        # Habitat) is treated as such by matplotlib
        learned_model_cloud[:, 1],
        c=learned_model_cloud[:, 2],
        alpha=0.3,
    )

    # Add hotspots
    for current_episode in range(episode_range):
        assert (
            detailed_stats[str(current_episode)]["target"]["primary_target_object"]
            == object_id
        ), "Episodes should all involve the same object"

        # === We only visualize successful jumps, i.e. that land on the object ===
        mask = detailed_stats[str(current_episode)]["motor_system"]["action_details"][
            "jump_successful"
        ]

        if len(mask) > 0:
            hotspot_locs = detailed_stats[str(current_episode)]["motor_system"][
                "action_details"
            ]["target_on_surface_model"]
            hotspot_locs = np.array(hotspot_locs)[mask]
            ax.scatter(
                hotspot_locs[:, 0],
                hotspot_locs[:, 2],
                hotspot_locs[:, 1],
                c="limegreen",
                alpha=0.9,
                s=200,
            )

    ax.set_aspect("equal")
    if view is not None:
        ax.view_init(view[0], view[1])

    if save_fig:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print("figure saved at " + save_path)
        plt.savefig(
            save_path,
            bbox_inches="tight",
            dpi=300,
        )
    else:
        plt.show()


def plot_evidence_transitions(
    episode,
    lm_stats,
    detection_fun,
    detection_params_dict,
    primary_target,
    color_mapping,
    stop_at_detected_new_object=True,
    save_fig_path=None,
):
    """Plot the change in an LM's evidence over time for different objects.

    The plot also shows the stepwise target that the learning module is receiving at
    any given time point.

    Currently this is mixed in with code for the detection of when the LM is on a
    new object, but this will be pulled out / refactored in the next PR (TODO) when
    adding the goal-state-generator class

    Args:
        episode: ?
        lm_stats: ?
        detection_fun: the detection function used to determine whether on a new
            object or not
        detection_params_dict: Dictionary of hyperparameters required by the
            detection function used.
        primary_target: ?
        color_mapping: Color association between object ID and color to make
            plot-interpetation easier; temporary
        detection_mapping: Color mapping associating types of detection (e.g. false
            positive, true positive, etc.) with particular colors.
        stop_at_detected_new_object: Given we currently don't detect new objects and
            initiate a corrective action, this is a hack to basically only look at the
            episode up until we've been off the primary target for a few consecutive
            steps; this makes the meaning of "false-positive", "true-positive", etc.
            much clearer, as ultimately our aim is to detect when we've fallen off the
            object we're trying to recognize, not detect every single time the
            stepwise target changes. Defaults to True.
        save_fig_path: ?. Defaults to None.
    """
    objects_list = lm_stats["evidences"][0].keys()

    stepwise_targets = np.array(lm_stats["stepwise_targets_list"])

    lm_processed = np.array(lm_stats["lm_processed_steps"])

    detection_cmapping = {
        "true_positive": "blue",
        "false_positive": "red",
        "false_negative": "grey",
        # True negative excluded because it is the default for a given step
    }

    _, ax = plt.subplots()

    processed_stepwise_targets = stepwise_targets[lm_processed]

    terminus_point = find_step_on_new_object(
        processed_stepwise_targets, primary_target, n_steps_off_primary_target=3
    )
    if terminus_point is None or not stop_at_detected_new_object:
        stop_point = len(lm_stats["evidences"])
    else:
        stop_point = terminus_point + 1

    # Log results like FP and TN
    detection_results = {
        "true_positive": 0,
        "false_positive": 0,
        "true_negative": 0,
        "false_negative": 0,
    }

    for current_obj in objects_list:
        max_ev_per_step = [0]

        for ii in range(stop_point):
            this_step_max = np.max(lm_stats["evidences"][ii][current_obj])
            max_ev_per_step.append(this_step_max)

            # If the current-object was the MLH on this step, perform
            # check for whether we're on a new object
            # Also need a minimum of two observations
            if current_obj == lm_stats["current_mlh"][ii]["graph_id"] and ii >= 1:
                if detection_fun(
                    max_ev_per_step,
                    **detection_params_dict,
                ):
                    # Currently focusing on movements off the primary target object
                    # TODO log these detections when this is part of the LM,
                    # and can then plot more naturally
                    if primary_target != processed_stepwise_targets[ii]:
                        detection = "true_positive"
                    else:
                        detection = "false_positive"

                    ax.scatter(
                        ii + 1,
                        this_step_max,
                        marker="x",
                        s=85,
                        color=detection_cmapping[detection],
                    )
                elif primary_target != processed_stepwise_targets[ii]:
                    detection = "false_negative"
                    ax.scatter(
                        ii + 1,
                        this_step_max,
                        marker="x",
                        s=85,
                        color=detection_cmapping[detection],
                    )
                else:
                    detection = "true_negative"
                    # Not plotted because the "default" result

                # Log results
                # NB these results are only calculated when the current object is the
                # MLH, i.e most of the outer-most loop does not contribute to these
                # results
                detection_results[detection] += 1

        ax.plot(
            range(stop_point + 1),
            max_ev_per_step,
            color=color_mapping[current_obj],
            alpha=0.5,
            label=current_obj,
        )

    ax.set_ylabel("Max Evidence")
    ax.set_xlabel("Step")

    # Add terminus line to plot
    if terminus_point is not None:
        ax.axvline(terminus_point + 1, color="grey", linewidth=3, alpha=0.7)

    ax.set_ylim(bottom=-2)
    ax2 = ax.twinx()

    # Plot the stepwise targets
    for ii in range(stop_point):
        ax2.bar(
            [float(ii) + 0.5],
            height=[-10],
            width=0.9,
            bottom=[-2.8],
            color=color_mapping[processed_stepwise_targets[ii]],
            linewidth=9,
            alpha=1.0,
        )

    ax2.set_ylabel("Stepwise Target", loc="bottom")
    ax2.set_ylim(bottom=-5, top=60)
    ax2.set_yticks([])

    ax.legend(prop={"size": 8})
    plt.title("Primary target : " + primary_target)

    if save_fig_path is not None:
        plt.savefig(
            save_fig_path + f"evidence_evolution_{episode}.png",
            bbox_inches="tight",
            dpi=300,
        )
