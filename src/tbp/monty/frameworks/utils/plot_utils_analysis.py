# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""A collection of plot utilities relying on seaborn and IPython.

Do not read too much into the "_analysis" suffix of this file. We needed separate
files to segment what is imported.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython import display
from matplotlib import animation

from tbp.monty.frameworks.utils.logging_utils import (
    check_detection_accuracy_at_step,
    check_rotation_accuracy,
)
from tbp.monty.frameworks.utils.plot_utils import mark_obs
from tbp.monty.frameworks.utils.plot_utils_dev import (
    format_ax,
    get_action_name,
    get_match_step,
    get_model_colors,
    get_model_id,
    get_search_positions,
    plot_previous_path,
    plot_search_displacements,
    set_target_text,
    show_previous_possible_paths_with_nodes,
    update_text,
)


def plot_rotation_stat_animation(detailed_stats, n_steps):
    """Create an animation of how rotation accuracy evolves over time.

    Note: you will need to install seaborn and jupyter notebook to use this function.
    """
    fig = plt.figure(figsize=(17, 5))
    ax = plt.subplot(1, 1, 1)
    sns.set(font_scale=1.5)

    def init():
        # avoid calling 0 twice
        pass

    def animate(frame_step, total_steps):
        step = total_steps - (frame_step + 1)
        rotation_stats = check_rotation_accuracy(detailed_stats, last_n_step=step + 1)
        ax.clear()
        h = sns.histplot(
            data=rotation_stats.sort_values("primary_performance"),
            x="primary_performance",
            hue="object",
            multiple="stack",
            palette="pastel",
        )
        plt.title(f"{step} steps before done")
        if len(h.get_xticks()) == 5:
            h.set_xticks(range(5))
            h.set_xticklabels(
                [
                    "in possible\nposes",
                    "correct",
                    "object not\nin matches",
                    "wrong rotation",
                    "not in possible\nposes",
                ]
            )
        #         else:
        #             h.set_xticks(range(5))
        # TODO: how to make the x axis stay the same always?
        h.set_ylim([0, len(detailed_stats.keys())])

    anim = animation.FuncAnimation(
        fig, animate, frames=n_steps, fargs=(n_steps,), interval=1000
    )
    video = anim.to_html5_video()
    html = display.HTML(video)
    display.display(html)
    plt.close()


def make_detection_stat_animation(detailed_stats, n_steps):
    """Display how detection accuracy evolves over time.

    Note: you will need to install seaborn to use this function.

    Returns:
        fig: Figure
        ax: Axis
        anim: Animation
    """
    fig = plt.figure(figsize=(17, 5))
    ax = plt.subplot(1, 1, 1)
    sns.set(font_scale=1.5)

    def init():
        # avoid calling 0 twice
        pass

    def animate(frame_step, total_steps):
        step = total_steps - (frame_step + 1)
        detection_stats = check_detection_accuracy_at_step(
            detailed_stats, last_n_step=step + 1
        )
        ax.clear()
        h = sns.histplot(
            data=detection_stats.sort_values("primary_performance"),
            x="primary_performance",
            hue="object",
            multiple="stack",
            palette="pastel",
        )
        plt.title(f"{step} steps before done")
        h.set_ylim([0, len(detailed_stats.keys())])

    anim = animation.FuncAnimation(
        fig, animate, frames=n_steps, fargs=(n_steps,), interval=1000
    )

    return fig, ax, anim


def plot_sample_animation(all_obs, patch_obs, viz_obs):
    """Plot video of sampled oservations."""
    fig = plt.figure(figsize=(12, 7))
    ax1 = fig.add_subplot(1, 3, 1)
    marked_obs = viz_obs[0].copy()
    # TODO update all plotting functions that use "marked-obs" to handle arbitrary
    # dimensions of the view-finder resolution, rather than being hardcoded to 64x64
    marked_obs[29:35, 29:35] = [0, 0, 255, 255]
    im1 = plt.imshow(marked_obs)
    ax1.set_xticks([]), ax1.set_yticks([])
    plt.title("Overview (Zoomed out)")
    ax2 = fig.add_subplot(1, 3, 2)
    im2 = plt.imshow(patch_obs[0])
    plt.title("Sensor View")
    ax2.set_xticks([]), ax2.set_yticks([])
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")

    num_steps = len(all_obs)
    plot_obs = all_obs[0]
    for obs in all_obs[1:]:
        # obj_obs = obs[np.where(obs[:, 3] > 0)]
        plot_obs = np.append(plot_obs, obs, axis=0)
    res = plot_obs.shape[0] // num_steps
    obj_obs = plot_obs[np.where(plot_obs[:res, 3] > 0)]  # & (plot_obs[:res, 2] < 0))

    scale_obs = plot_obs[np.where(plot_obs[:, 3] > 0)]
    p1 = ax3.scatter(
        -obj_obs[:, 1],
        obj_obs[:, 0],
        obj_obs[:, 2],
        c=obj_obs[:, 2],
        vmin=min(scale_obs[:, 2]),
        vmax=max(scale_obs[:, 2]),
    )

    ax3.set_xticks([]), ax3.set_yticks([]), ax3.set_zticks([])
    ax3.set_xlabel("x", labelpad=-10)
    ax3.set_ylabel("y", labelpad=-10)
    ax3.set_zlabel("z", labelpad=-10)

    plot_zoom = 0.07
    means = np.mean(plot_obs, axis=0)
    ax3.set_xlim([-means[1] - plot_zoom, -means[1] + plot_zoom])
    ax3.set_ylim([means[0] - plot_zoom, means[0] + plot_zoom])
    ax3.set_zlim([means[2] - plot_zoom, means[2] + plot_zoom])
    ax3.view_init(110, 0)

    def init():
        # avoid calling 0 twice
        pass

    def animate(i):
        marked_obs = viz_obs[i].copy()
        marked_obs[29:35, 29:35] = [0, 0, 255, 255]
        im1.set_array(marked_obs)
        im2.set_array(patch_obs[i])

        point_idx = int((i + 1) * res)
        obj_obs = plot_obs[
            np.where(plot_obs[:point_idx, 3] > 0)  # & (plot_obs[:point_idx, 2] < 0)
        ]
        p1._offsets3d = (-obj_obs[:, 1], obj_obs[:, 0], obj_obs[:, 2])
        p1.set_array(obj_obs[:, 2])

        return (ax1,)

    anim = animation.FuncAnimation(fig, animate, frames=len(viz_obs), init_func=init)
    video = anim.to_html5_video()
    html = display.HTML(video)
    display.display(html)
    plt.close()


def plot_sample_animation_multiobj(
    patch_obs, viz_obs, semantic_obs, primary_target="", save_bool=False
):
    """Simplified video of sampled oservations.

    This video supports labelling of targets when there are multiple objects in the
    environment.
    """
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    marked_obs = viz_obs[0].copy()
    marked_obs[29:35, 29:35] = [0, 0, 255, 255]
    im1 = plt.imshow(marked_obs)
    ax1.set_xticks([]), ax1.set_yticks([])
    plt.title("Overview (Zoomed out)")
    ax2 = fig.add_subplot(1, 2, 2)
    im2 = plt.imshow(patch_obs[0])
    ax2.set_xticks([]), ax2.set_yticks([])

    def init():
        # avoid calling 0 twice
        pass

    def animate(i):
        marked_obs = viz_obs[i].copy()
        marked_obs[29:35, 29:35] = [0, 0, 255, 255]
        im1.set_array(marked_obs)
        im2.set_array(patch_obs[i])
        plt.title(
            "primary_target: "
            + primary_target
            + "\nstepwise_target: "
            + semantic_obs[i]
        )

        return (ax1,)

    anim = animation.FuncAnimation(fig, animate, frames=len(viz_obs), init_func=init)
    video = anim.to_html5_video()
    html = display.HTML(video)
    display.display(html)

    if save_bool:
        anim.save(
            "viewfinder_gif.gif",
            writer="imagemagick",
            dpi=300,
        )
    plt.close()


def plot_detection_animation(
    all_obs, patch_obs, viz_obs, path_matches, model, model_name
):
    """Plot video of object detection using displacements."""
    fig = plt.figure(figsize=(12, 7))
    ax1 = fig.add_subplot(1, 4, 1)
    marked_obs = viz_obs[0].copy()
    marked_obs[29:35, 29:35] = [0, 0, 255, 255]
    im1 = plt.imshow(marked_obs)
    ax1.set_xticks([]), ax1.set_yticks([])
    plt.title("Overview (Zoomed out)")
    ax2 = fig.add_subplot(1, 4, 2)
    im2 = plt.imshow(patch_obs[0])
    plt.title("Sensor View")
    ax2.set_xticks([]), ax2.set_yticks([])
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")

    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    num_points = model.x.shape[0]
    colors = np.zeros(num_points)

    num_steps = len(all_obs)
    plot_obs = all_obs[0]
    for obs in all_obs[1:]:
        plot_obs = np.append(plot_obs, obs, axis=0)
    res = plot_obs.shape[0] // num_steps
    obj_obs = plot_obs[np.where(plot_obs[:res, 3] > 0)]
    # For scaling the plot the same way as the model graph
    ax3.scatter(
        -model.pos[:, 1], model.pos[:, 0], model.pos[:, 2], c="white", alpha=0.1
    )
    p1 = ax3.scatter(
        -obj_obs[:, 1],
        obj_obs[:, 0],
        obj_obs[:, 2],
        c=obj_obs[:, 2],
        vmin=min(plot_obs[:, 2]),
        vmax=max(plot_obs[:, 2]),
    )
    plt.title("Patch Observations")

    ax3.set_xticks([]), ax3.set_yticks([]), ax3.set_zticks([])
    ax3.set_xlabel("x", labelpad=-10)
    ax3.set_ylabel("y", labelpad=-10)
    ax3.set_zlabel("z", labelpad=-10)
    ax3.view_init(110, 0)

    colors = np.zeros(num_points)
    for p in path_matches[0][model_name]:
        colors[int(p)] = 1
    p2 = ax4.scatter(
        -model.pos[:, 1],
        model.pos[:, 0],
        model.pos[:, 2],
        c=colors,
        vmin=0,
        cmap="RdYlGn",
    )
    plt.title("Model Matches")
    ax4.view_init(110, 0)
    ax4.set_xticks([]), ax4.set_yticks([]), ax4.set_zticks([])

    def init():
        # avoid calling 0 twice
        pass

    def animate(i):
        marked_obs = viz_obs[i].copy()
        marked_obs[29:35, 29:35] = [0, 0, 255, 255]
        im1.set_array(marked_obs)
        im2.set_array(patch_obs[i])

        point_idx = int((i + 1) * res)
        obj_obs = plot_obs[np.where(plot_obs[:point_idx, 3] > 0)]
        p1._offsets3d = (-obj_obs[:, 1], obj_obs[:, 0], obj_obs[:, 2])
        p1.set_array(obj_obs[:, 2])

        colors = np.zeros(num_points)
        if i == 0:
            for surface_normal, p in enumerate(path_matches[i][model_name]):
                colors[int(p)] = surface_normal / (i + 1) + 0.5
        elif len(path_matches[i][model_name]) > 0:
            for surface_normal, p in enumerate(path_matches[i][model_name][0]):
                colors[int(p)] = surface_normal / (i + 1) + 0.5
        else:
            pass
        p2.set_array(colors)

        return (ax1,)

    anim = animation.FuncAnimation(fig, animate, frames=len(viz_obs), init_func=init)
    video = anim.to_html5_video()
    html = display.HTML(video)
    display.display(html)
    plt.close()


def plot_feature_matching_animation(
    stats,
    lm_models,
    episode,
    objects,
    lm_id="LM_0",
    lm_num=0,
    sm_id_patch="SM_0",
    sm_id_vis="SM_1",
    object_names=None,
    show_num_pos=None,
    show_path=False,
    ax_range=None,
    rotate=True,
    show_num_steps=None,
):
    """Plot video of object detection using features at locations.

    TODO: use SM_0 obs to know which raw obs were on the object and show the
    correct corresponding images.
    """
    epoch = stats[str(episode)][lm_id]["train_epochs"]
    model_id = get_model_id(epoch, stats[str(episode)][lm_id]["mode"])
    # num_steps = len(stats[str(episode)][lm_id]["possible_matches"])
    # num_steps = len(stats[str(episode)][sm_id_patch]["raw_observations"])
    target = stats[str(episode)][lm_id]["target"]

    processed_obs_ids = []
    for i, po in enumerate(stats[str(episode)][sm_id_patch]["processed_observations"]):
        # Use this extra if statement if plotting observations from surface-agent
        # sensor run. TODO: make this more general.
        # if (po is not None) and ((i + 1) % 4 == 0):
        if po is not None:
            processed_obs_ids.append(i)

    if show_num_steps is None:
        num_steps = len(processed_obs_ids)
    else:
        num_steps = show_num_steps

    if object_names is None:
        object_names = objects

    fig = plt.figure(figsize=(10, 5))
    fig.tight_layout()
    vis_ax = fig.add_subplot(1, len(objects) + 1, 1)
    vis_ax.axis("off")

    axes = []
    text_axes = []
    all_colors = []
    all_model_pos = np.array([0, 0, 0])

    for i, object_n in enumerate(objects):
        # epoch -1 because we want the model at the end of the previous epoch
        model_pos = lm_models[model_id][lm_num][object_n].pos
        all_model_pos = np.vstack([all_model_pos, model_pos.numpy()])
        # Add axes for 3d model plots
        ax = fig.add_subplot(1, len(objects) + 1, i + 2, projection="3d")
        ax.set_title(object_names[i])
        ax.text(0, 0, 0, target["object"])
        axes.append(ax)
        colors = np.ones(model_pos.shape[0])
        all_colors.append(colors)
        # Add axes for num path and pose text
        text_ax = fig.add_subplot(1, len(objects) + 1, i + 2)
        text_ax.axis("off")
        text = text_ax.text(0, 0.0, "")
        text_axes.append(text)

    ax = fig.add_subplot(2, 1, 2)
    set_target_text(ax, target, len(objects))

    def animate(plot_step):
        is_match_step = bool(plot_step % 2)
        step = plot_step // 2
        # get observations for this step
        lm_obs_id = processed_obs_ids[step]  # get id of obs that was sent to LM
        patch_raw_obs = stats[str(episode)][sm_id_patch]["raw_observations"][lm_obs_id]
        patch_processed_obs = stats[str(episode)][sm_id_patch][
            "processed_observations"
        ][lm_obs_id]
        vis_obs = stats[str(episode)][sm_id_vis]["raw_observations"][lm_obs_id]
        obs_on_object = bool(patch_processed_obs["features"]["on_object"])
        # Show sensor observations on left
        marked_obs = mark_obs(
            vis_obs,
            patch_raw_obs,
        )
        vis_ax.imshow(marked_obs / 255.0)
        # Set title above vis obs to action or matching step
        action_name = get_action_name(
            stats[str(episode)]["motor_system"]["action_sequence"],
            lm_obs_id,
            is_match_step,
            obs_on_object,
        )
        vis_ax.set_title(action_name)
        if obs_on_object:
            all_on_obj = np.array(
                [
                    stats[str(episode)][sm_id_patch]["processed_observations"][i][
                        "features"
                    ]["on_object"]
                    for i in processed_obs_ids
                ]
            )
            steps_on_obj = sum(all_on_obj[:step])
            step = int(steps_on_obj)

            displacement = stats[str(episode)][lm_id]["displacement"][step]
            # Show matching process for each object in objects
            for i, object_n in enumerate(objects):
                # Starting from scratch at each step bc it seemed easier for now
                axes[i].clear()
                # Load model from end of previous epoch
                model_pos = lm_models[model_id][lm_num][object_n].pos
                all_colors[i] = get_model_colors(
                    model_pos.shape[0], step, is_match_step
                )
                matches_at_step = get_match_step(is_match_step, step)

                current_matches = stats[str(episode)][lm_id]["possible_matches"][
                    matches_at_step
                ]
                num_possible_paths = 0
                all_poses = np.array([[0, 0, 0]])
                if object_n in current_matches:
                    possible_paths = stats[str(episode)][lm_id]["possible_paths"][
                        matches_at_step
                    ][object_n]
                    num_possible_paths = len(possible_paths)
                    for path in range(num_possible_paths):
                        current_path = possible_paths[path]
                        possible_poses = stats[str(episode)][lm_id]["possible_poses"][
                            matches_at_step
                        ][object_n][path]

                        all_poses = np.vstack([all_poses, np.array(possible_poses)])
                        start_node = current_path[-1]
                        if is_match_step:
                            # Show current possible location in green
                            axes[i].scatter(
                                start_node[0],
                                start_node[1],
                                start_node[2],
                                s=40,
                                c="green",
                                vmin=0,
                            )
                        elif step > 0:
                            search_positions = get_search_positions(
                                start_node,
                                possible_poses[:show_num_pos],
                                displacement,
                            )
                            plot_search_displacements(
                                axes[i], search_positions, start_node
                            )
                        # Plot Path
                        if show_path:
                            plot_previous_path(axes[i], current_path, step)

                all_poses = all_poses[1:]  # exclude first default element

                axes[i].scatter(
                    model_pos[:, 0],
                    model_pos[:, 1],
                    model_pos[:, 2],
                    c=all_colors[i],
                    alpha=0.5,
                    vmin=0,
                    vmax=1,
                    s=2,
                    cmap="RdYlGn",
                )
                axes[i].set_title(object_names[i])

                if is_match_step:
                    unique_poses = np.unique(all_poses, axis=0)
                    update_text(text_axes[i], num_possible_paths, unique_poses)
                    show_previous_possible_paths_with_nodes(
                        axes[i],
                        stats[str(episode)][lm_id]["possible_paths"],
                        step + 1,
                        object_n,
                    )
                else:
                    show_previous_possible_paths_with_nodes(
                        axes[i],
                        stats[str(episode)][lm_id]["possible_paths"],
                        step,
                        object_n,
                    )

            for ax in axes:
                format_ax(ax, all_model_pos, ax_range, rotate, plot_step)
            fig.suptitle(f"Step {step}", y=0.85)
        return (vis_ax,)

    anim = animation.FuncAnimation(fig, animate, frames=num_steps * 2, interval=1000)
    video = anim.to_html5_video()
    html = display.HTML(video)
    display.display(html)
    plt.close()


def plot_detection_stat_animation(detailed_stats, n_steps):
    """Display detection stat animation in jupyter notebook."""
    _, _, anim = make_detection_stat_animation(detailed_stats, n_steps)
    video = anim.to_html5_video()
    html = display.HTML(video)
    display.display(html)
    plt.close()


def save_detection_stat_animation(detailed_stats, n_steps, output_file):
    """Save detection stat animation to output_file."""
    _, _, anim = make_detection_stat_animation(detailed_stats, n_steps)
    if not output_file.endswith(".mp4"):
        output_file = output_file + ".mp4"
    writervideo = anim.FFMpegWriter(fps=60)
    anim.save(output_file, writer=writervideo)
    print(f"Animation saved to {output_file}")
