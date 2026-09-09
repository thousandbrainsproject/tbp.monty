# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb

from analysis.views import Points

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "CURRENT_CHAIN",
    "MODELS_DIR",
    "LearnedObject",
    "child_labels",
    "decode_object_ids",
    "default_model",
    "load_learned_objects",
    "object_id_feature",
    "resolve_model_path",
]

# Where pretraining runs write their models: <run_name>/pretrained/model.pt.
MODELS_DIR = (
    Path(
        os.environ.get("MONTY_MODELS", "~/tbp/results/monty/pretrained_models")
    ).expanduser()
    / "my_trained_models"
)

# The pretraining chain being trained now, first stage first (see
# conf/experiment/comp/). Each stage's model holds every earlier stage's
# objects; the last two are alternative third stages, the single-object
# cube_tbp one preferred when both exist.
CURRENT_CHAIN = (
    "supervised_pre_training_cube_cylinder_3d_children_mujoco",
    "supervised_pre_training_cube_cylinder_2d_children_mujoco",
    "supervised_pre_training_cube_cylinder_comp_models_mujoco",
    "supervised_pre_training_cube_tbp_comp_models_mujoco",
)


def default_model() -> Path:
    """The model to inspect when none is named: the chain's newest trained stage.

    Returns:
        The ``model.pt`` of the last stage of ``CURRENT_CHAIN`` that has one.

    Raises:
        FileNotFoundError: If no stage has been trained yet.
    """
    for run_name in reversed(CURRENT_CHAIN):
        path = MODELS_DIR / run_name / "pretrained" / "model.pt"
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"no stage of {CURRENT_CHAIN} is trained under {MODELS_DIR}"
    )


def resolve_model_path(model: str | os.PathLike) -> Path:
    """Turn a pretraining run name, run directory, or model file into the model file.

    Args:
        model: A ``model.pt`` path, a directory holding ``pretrained/model.pt``
            (or ``model.pt`` directly), or a run name under ``MODELS_DIR``.

    Returns:
        The path of the ``model.pt`` file.

    Raises:
        FileNotFoundError: If none of the candidate locations exist.
    """
    candidates = [Path(model).expanduser(), MODELS_DIR / str(model)]
    for base in candidates:
        for path in (base, base / "model.pt", base / "pretrained" / "model.pt"):
            if path.is_file():
                return path
    raise FileNotFoundError(f"no model.pt for {model!r} (looked under {candidates})")


@dataclass(frozen=True)
class LearnedObject:
    """One learned object graph: the nodes an LM stored for one input channel.

    Attributes:
        lm: The learning module that learned it, e.g. ``"LM_0"``.
        object_id: The learned object's name, e.g. ``"001_cube"``.
        channel: The input channel the nodes came from -- a sensor patch
            (``"patch_0"``) or a lower-level LM (``"learning_module_0"``).
        points: The node locations, in the model's reference frame, with one
            feature array per feature the graph stores (``pose_vectors`` is
            ``(N, 9)``: the surface normal followed by two tangent
            directions; ``hsv`` is ``(N, 3)``; scalars are ``(N,)``).
    """

    lm: str
    object_id: str
    channel: str
    points: Points

    @property
    def normals(self) -> np.ndarray:
        """The ``(N, 3)`` surface normal at each node."""
        return self.points["pose_vectors"][:, :3]

    @property
    def is_2d(self) -> bool:
        """Whether the nodes came from a 2D (edge-based) sensor module."""
        return "edge_strength" in self.points.features

    @property
    def morphology(self) -> np.ndarray:
        """The ``(N, 3)`` unit vector that describes each node's local shape.

        The surface normal (first pose vector) for 3D channels; for 2D
        channels, whose first pose vector is just the out-of-plane normal,
        the oriented edge direction stored as the second pose vector.
        """
        pose_vectors = self.points["pose_vectors"]
        return pose_vectors[:, 3:6] if self.is_2d else pose_vectors[:, :3]

    @property
    def colors(self) -> np.ndarray:
        """Each node's stored color as ``(N, 3)`` RGB in [0, 1].

        Gray when the graph stores no color (nodes learned from another LM).
        """
        if "hsv" in self.points.features:
            return hsv_to_rgb(np.clip(self.points["hsv"], 0.0, 1.0))
        return np.full((len(self.points), 3), 0.5)

    def __len__(self) -> int:
        return len(self.points)


def load_learned_objects(
    model: str | os.PathLike,
) -> dict[str, dict[str, dict[str, LearnedObject]]]:
    """Read every learned object graph out of a pretrained model file.

    Args:
        model: See :func:`resolve_model_path`.

    Returns:
        Nested by learning module (``"LM_<n>"``), then object id, then input
        channel.
    """
    state = torch.load(resolve_model_path(model), weights_only=False)
    learned: dict[str, dict[str, dict[str, LearnedObject]]] = {}
    for lm_index, lm_state in state["lm_dict"].items():
        lm = f"LM_{lm_index}"
        learned[lm] = {}
        for object_id, channels in lm_state["graph_memory"].items():
            learned[lm][object_id] = {
                channel: LearnedObject(lm, object_id, channel, _graph_points(graph))
                for channel, graph in channels.items()
            }
    return learned


def _graph_points(graph) -> Points:
    # A GraphObjectModel stores every feature in one (N, F) matrix, with
    # feature_mapping giving each feature's column span.
    x = np.asarray(graph.x)
    features = {}
    for name, (start, stop) in graph.feature_mapping.items():
        column = x[:, start:stop]
        features[name] = column[:, 0] if stop - start == 1 else column
    return Points(np.asarray(graph.pos, dtype=float), features)


def object_id_feature(object_id: str) -> int:
    """The number an LM stores for a child object it recognized.

    Mirrors ``EvidenceGraphLM._object_id_to_features``: the sum of the
    name's character codes, so it is not unique in general.

    Returns:
        The feature value.
    """
    return sum(ord(c) for c in object_id)


def decode_object_ids(values: np.ndarray, candidates: Iterable[str]) -> list[str]:
    """Name the child objects behind an ``object_id`` feature column.

    Args:
        values: The ``(N,)`` stored ``object_id`` feature values.
        candidates: The object names the child LM knows (its graph ids).

    Returns:
        Per value, the candidate whose feature value matches, ``"a|b"`` when
        several collide, or the number itself when none matches.
    """
    by_value: dict[int, list[str]] = {}
    for name in candidates:
        by_value.setdefault(object_id_feature(name), []).append(name)
    return ["|".join(by_value.get(round(v), [])) or f"{round(v)}" for v in values]


def child_labels(
    channels: dict[str, LearnedObject],
    learned: dict[str, dict[str, dict[str, LearnedObject]]],
) -> dict[str, list[str]]:
    """Label the nodes an object learned from lower-level learning modules.

    Args:
        channels: One object's graphs, one per input channel.
        learned: Everything the model learned, to name child objects by.

    Returns:
        Per ``learning_module_<k>`` channel, one ``"<channel>: <child>"``
        label per node.
    """
    labels = {}
    for channel, obj in channels.items():
        if channel.startswith("learning_module"):
            child_lm = "LM_" + channel.rsplit("_", 1)[-1]
            names = decode_object_ids(
                obj.points["object_id"], learned.get(child_lm, {})
            )
            labels[channel] = [f"{channel}: {name}" for name in names]
    return labels
