# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# ruff: noqa: DOC201,DOC501
"""Demonstrate feature-based graph-mismatch goals on compositional models.

Loads the pretrained cube/cylinder compositional models (3-LM MuJoCo experiment)
and runs the EvidenceGoalGenerator's graph-mismatch computation on the parent
LM's (LM 2's) graphs. The tbp and numenta variants of each parent object are
spatially near-identical (the logos are flat stickers), so Euclidean distance
cannot distinguish them; the GSG should instead detect the mismatching
object-ID features on the logo input channel and propose a goal at the center
of the logo.

Usage:
    python analysis/scripts/compositional_graph_mismatch_demo.py [MODEL_PATH] [options]

MODEL_PATH points at a trained model: either model.pt itself, its pretrained
directory, or the experiment directory containing it. Defaults to the
supervised_pre_training_cube_cylinder_comp_models_mujoco experiment in
~/tbp/results/monty/pretrained_models/my_trained_models.

Options:
    --lm ID         Which learning module's models to compare (default: 2).
    --output PATH   Where to save the figure
                    (default: ~/Desktop/graph_mismatch_demo.png).
    --show          Also open an interactive window.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.goal_generation import EvidenceGoalGenerator

DEFAULT_MODEL_PATH = (
    "~/tbp/results/monty/pretrained_models/my_trained_models/"
    "supervised_pre_training_cube_cylinder_comp_models_mujoco"
)

# Pairs of parent objects that only differ in the logo placed on them.
OBJECT_PAIRS = [
    ("002_cube_tbp", "004_cube_numenta"),
    ("012_cylinder_tbp_horz", "014_cylinder_numenta_horz"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_path",
        type=Path,
        nargs="?",
        default=Path(DEFAULT_MODEL_PATH),
        help="model.pt, its pretrained directory, or the experiment directory",
    )
    parser.add_argument("--lm", type=int, default=2, help="LM whose models to use")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("~/Desktop/graph_mismatch_demo.png"),
        help="where to save the figure",
    )
    parser.add_argument("--show", action="store_true", help="open a window")
    return parser.parse_args()


def resolve_checkpoint(model_path: Path) -> Path:
    """Resolve model.pt from a checkpoint, pretrained, or experiment path."""
    path = model_path.expanduser()
    for candidate in (path, path / "model.pt", path / "pretrained" / "model.pt"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Model checkpoint not found under: {path}")


def load_lm_memory(checkpoint: Path, lm_id: int) -> tuple[dict, dict[int, str]]:
    """Load one LM's graph memory and a decoder for object_id feature values."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    lm_dict = state["lm_dict"]
    lm_key = lm_id if lm_id in lm_dict else str(lm_id)
    memory = lm_dict[lm_key]["graph_memory"]

    # object_id features are encoded as the sum of the character codes of the
    # object's name, so every object name in the checkpoint decodes one ID
    id_names = {}
    for lm_state in lm_dict.values():
        for object_name in lm_state.get("graph_memory", {}):
            id_names[sum(ord(c) for c in object_name)] = object_name
    return memory, id_names


class FakeParentLM:
    """Serve a checkpoint's graph memory through the parent-LM interface.

    Implements just the methods the EvidenceGoalGenerator uses, with the top
    and second most-likely hypotheses fixed to a chosen object pair (identity
    rotations at the origin, i.e. the graphs are compared in the object-centric
    frames they were learned in).
    """

    def __init__(self, memory: dict, top_id: str, second_id: str):
        self.memory = memory
        self.top_id = top_id
        self.second_id = second_id
        self.learning_module_id = "demo_lm"
        sensor_channels = [
            channel for channel in memory[top_id] if channel.startswith("patch")
        ]
        self.buffer = SimpleNamespace(
            get_first_sensory_input_channel=lambda: sensor_channels[0],
            update_stats=lambda *_args, **_kwargs: None,
        )

    def get_top_two_mlh_ids(self):
        return self.top_id, self.second_id

    def get_mlh_for_object(self, object_id):
        return {
            "graph_id": object_id,
            "location": np.zeros(3),
            "rotation": Rotation.identity(),
        }

    def _get_current_mlh(self):
        return self.get_mlh_for_object(self.top_id)

    def get_graph(self, graph_id, input_channel=None):
        if input_channel is None:
            return self.memory[graph_id]
        return self.memory[graph_id][input_channel]

    def get_input_channels_in_graph(self, graph_id):
        return list(self.memory[graph_id].keys())


def run_pair(memory: dict, top_id: str, second_id: str, id_names: dict[int, str]):
    """Run the graph-mismatch computation for one object pair.

    Returns:
        Tuple of (winning input channel, target node id, target location,
        target surface normal).
    """
    # The learned graphs of each pair contain a few spurious points whose
    # nearest-neighbor separation is slightly above 1cm (up to ~1.9cm), caused
    # by uneven sampling during learning rather than a real shape difference.
    # Raising the spatial threshold to 2cm routes these near-identical shapes
    # to the feature-based path.
    gsg = EvidenceGoalGenerator(
        min_post_goal_success_steps=5,
        feature_mismatch_distance_threshold=0.02,
    )
    gsg.parent_lm = FakeParentLM(memory, top_id, second_id)
    gsg.focus_on_pose = False
    ctx = RuntimeContext(rng=np.random.RandomState(0))

    mismatch = gsg._compute_graph_mismatch(ctx)
    if mismatch is None:
        raise RuntimeError(
            f"No mismatch found between {top_id} and {second_id}; expected an "
            "object-ID mismatch on the logo channel."
        )
    channel, target_loc_id = mismatch
    target_info = gsg._get_target_loc_info(target_loc_id, channel)
    target_loc = np.asarray(target_info["target_loc"])
    target_normal = np.asarray(target_info["target_surface_normal"])

    top_graph = memory[top_id][channel]
    fm = top_graph.feature_mapping
    if "object_id" in fm:
        raw_id = float(top_graph.x[target_loc_id, fm["object_id"][0]])
        stored = id_names.get(int(raw_id), raw_id)
        print(
            f"{top_id} vs {second_id}: goal on channel '{channel}' at node "
            f"{target_loc_id}, location {np.round(target_loc, 4)} "
            f"(stored child object: {stored})"
        )
    else:
        print(
            f"{top_id} vs {second_id}: goal on channel '{channel}' at node "
            f"{target_loc_id}, location {np.round(target_loc, 4)}"
        )
    return channel, target_loc_id, target_loc, target_normal


def verify_goal_on_logo(memory: dict, top_id: str, channel: str, target_loc):
    """Check the proposed goal sits within the logo (LM-1 input) region."""
    logo_positions = np.asarray(memory[top_id]["learning_module_1"].pos)
    dist_to_logo = np.linalg.norm(logo_positions - target_loc, axis=1).min()
    assert channel == "learning_module_1", (
        f"Expected the goal to come from the logo channel, got '{channel}'"
    )
    assert dist_to_logo < 0.01, (
        f"Goal is {dist_to_logo * 100:.1f}cm from the nearest logo node"
    )
    print(
        f"  -> verified: goal is on the logo region "
        f"(distance to nearest logo node: {dist_to_logo * 1000:.2f}mm)"
    )


def plot_pair(ax, memory: dict, top_id: str, target_loc, target_normal):
    """Plot the parent object's channels and the proposed goal."""
    channels = memory[top_id]
    sensor_channel = next(c for c in channels if c.startswith("patch"))
    patch_pos = np.asarray(channels[sensor_channel].pos)
    ax.scatter(
        *patch_pos.T, s=2, c="lightgray", alpha=0.5, label=sensor_channel
    )
    channel_styles = (
        ("learning_module_0", "#00a0df", 3, 0.35),
        ("learning_module_1", "#e8710a", 30, 1.0),
    )
    for channel, color, size, alpha in channel_styles:
        if channel in channels:
            pos = np.asarray(channels[channel].pos)
            ax.scatter(*pos.T, s=size, c=color, alpha=alpha, label=channel)
    ax.scatter(
        *np.atleast_2d(target_loc).T,
        s=250,
        c="red",
        marker="*",
        edgecolors="black",
        label="proposed goal",
        zorder=10,
    )
    ax.quiver(
        *target_loc, *(target_normal * 0.03), color="red", linewidth=2, zorder=10
    )
    ax.set_title(top_id)
    ax.set_box_aspect((1, 1, 1))
    # Look down at the top face, where the logos sit on these objects
    ax.view_init(elev=50, azim=-55)
    limits = np.array([patch_pos.min(axis=0), patch_pos.max(axis=0)])
    center, half_span = limits.mean(axis=0), (limits[1] - limits[0]).max() / 2
    ax.set_xlim(center[0] - half_span, center[0] + half_span)
    ax.set_ylim(center[1] - half_span, center[1] + half_span)
    ax.set_zlim(center[2] - half_span, center[2] + half_span)
    ax.legend(loc="upper left", fontsize=7)


def main():
    args = parse_args()
    checkpoint = resolve_checkpoint(args.model_path)
    memory, id_names = load_lm_memory(checkpoint, args.lm)
    print(f"Loaded LM_{args.lm} graph memory from {checkpoint}")
    print(f"Objects: {list(memory.keys())}\n")

    figure, axes = plt.subplots(
        1, len(OBJECT_PAIRS), figsize=(7 * len(OBJECT_PAIRS), 7),
        subplot_kw={"projection": "3d"},
    )
    axes = np.atleast_1d(axes)

    for ax, (top_id, second_id) in zip(axes, OBJECT_PAIRS):
        channel, _, target_loc, target_normal = run_pair(
            memory, top_id, second_id, id_names
        )
        verify_goal_on_logo(memory, top_id, channel, target_loc)
        plot_pair(ax, memory, top_id, target_loc, target_normal)

    figure.suptitle(
        "Hypothesis-testing goals from object-ID feature mismatch\n"
        "(red star: proposed goal; arrow: surface normal used to approach)"
    )
    figure.tight_layout()
    output = args.output.expanduser()
    figure.savefig(output, dpi=150)
    print(f"\nSaved figure to {output}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
