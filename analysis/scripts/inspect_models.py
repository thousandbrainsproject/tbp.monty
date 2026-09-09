# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Inspect a pretrained model's object graphs interactively, with vedo.

One learning module at a time: a window with one renderer per learned
object, orbitable with the mouse. Nodes learned from a sensor patch show
their stored color (or a chosen feature); nodes learned from a lower-level
LM are spheres colored per child object. Every point type -- each sensor
channel and each ``channel: child`` combination -- has a checkbox button in
the first renderer that shows or hides it in every renderer; ``q`` closes.

Needs the ``viz`` extra (``uv sync --extra viz``). Run from the repo root,
e.g. ``python -m analysis.scripts.inspect_models --lm LM_2`` for the current
chain's newest trained stage (see ``analysis.models.CURRENT_CHAIN``), or
name a model: a ``model.pt`` path, a run directory, or a run name under
``MODELS_DIR``. ``--screenshot`` renders offscreen to a file instead.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import cycle
from typing import TYPE_CHECKING

import numpy as np
import vedo

from analysis.models import (
    LearnedObject,
    child_labels,
    default_model,
    load_learned_objects,
    resolve_model_path,
)

if TYPE_CHECKING:
    import os
    from collections.abc import Callable, Sequence

# Color nodes by their stored color rather than by a feature.
STORED_COLOR = "color"
# The learning module shown unless another is asked for: the top of the 3LM
# heterarchy, whose compositional graphs are the ones to inspect.
DEFAULT_LM = "LM_2"
# Bold colors for child objects; deliberately absent from the objects
# themselves so they cannot be confused with stored patch colors.
CHILD_COLORS = ("#FF2DAF", "#E8000B", "#00B140", "#FFC20A", "#7B2CBF", "#00A6D6")
# Checkbox button layout, in normalized window coordinates of the first renderer.
BUTTON_X = 0.02
BUTTON_TOP = 0.95
BUTTON_STEP = 0.05


def inspect_model(
    model: str | os.PathLike | None = None,
    lm: str = DEFAULT_LM,
    objects: Sequence[str] | None = None,
    color_by: str = STORED_COLOR,
    point_size: int = 6,
    screenshot: os.PathLike | None = None,
) -> None:
    """Open one learning module's object graphs in a vedo window.

    Args:
        model: See :func:`analysis.models.resolve_model_path`; the newest
            trained stage of the current chain when None.
        lm: The learning module to show (``"LM_0"`` ...); ``DEFAULT_LM``
            unless given, falling back to the first module that learned
            anything when the chosen one is empty.
        objects: The object ids to show; all when None.
        color_by: ``"color"`` for the nodes' stored color, else the name of a
            stored feature to color sensor-learned nodes by (a vector
            feature's first component).
        point_size: Screen-space radius of the sensor-learned nodes.
        screenshot: Render offscreen and save to this path instead of
            opening a window.

    Raises:
        ValueError: If the model holds no objects for the requested module.
    """
    model_path = default_model() if model is None else resolve_model_path(model)
    learned = load_learned_objects(model_path)
    if not learned.get(lm):
        # Fall back to the first module that learned anything (e.g. an
        # earlier pretraining stage, where the top-level LM is still empty).
        lm = next((name for name, objs in learned.items() if objs), lm)
    by_object = learned.get(lm, {})
    if objects:
        by_object = {name: by_object[name] for name in objects if name in by_object}
    if not by_object:
        raise ValueError(f"{model_path} holds no objects for {lm!r}")

    # Every child-object label gets one color across the window.
    labels = sorted(
        {
            label
            for channels in by_object.values()
            for node_labels in child_labels(channels, learned).values()
            for label in node_labels
        }
    )
    label_colors = dict(zip(labels, cycle(CHILD_COLORS)))

    plotter = vedo.Plotter(
        N=len(by_object),
        title=f"{model_path.parent.parent.name} -- {lm}",
        axes=1,
        offscreen=screenshot is not None,
    )
    # Per point type, its actors across every renderer, so one checkbox
    # shows or hides the type everywhere.
    actors_by_type: dict[str, list] = defaultdict(list)
    for at, (object_id, channels) in enumerate(by_object.items()):
        actors = _object_actors(channels, learned, label_colors, color_by, point_size)
        for point_type, type_actors in actors.items():
            actors_by_type[point_type].extend(type_actors)
        total = sum(len(c) for c in channels.values())
        # Not interactive yet: show() would otherwise block on the first
        # renderer, before the other objects and the checkboxes exist.
        plotter.at(at).show(
            *(actor for type_actors in actors.values() for actor in type_actors),
            f"{object_id} ({total} nodes)",
            resetcam=True,
            interactive=False,
        )

    _add_checkboxes(plotter, actors_by_type)
    plotter.render()
    if screenshot is not None:
        plotter.screenshot(str(screenshot))
        print(f"Saved {screenshot}")
    else:
        plotter.interactive()
    plotter.close()


def _add_checkboxes(
    plotter: vedo.Plotter, actors_by_type: dict[str, list]
) -> dict[str, Callable[[], None]]:
    """Add one show/hide checkbox button per point type to the first renderer.

    Returns:
        Per point type, a function that shows or hides it everywhere.
    """
    toggles: dict[str, Callable[[], None]] = {}
    plotter.at(0)
    for row, (point_type, actors) in enumerate(actors_by_type.items()):
        shown = bool(actors) and bool(actors[0].actor.GetVisibility())
        states = [f"[x] {point_type}", f"[ ] {point_type}"]
        if not shown:
            states.reverse()
        # The button's pick event calls the function it was created with, so
        # the toggle is defined first and reaches its button through `holder`.
        holder: dict[str, vedo.addons.Button] = {}

        def toggle(*_args, actors=actors, holder=holder) -> None:
            for actor in actors:
                actor.toggle()
            holder["button"].switch()
            plotter.render()

        holder["button"] = _add_button(
            plotter,
            toggle,
            states=states,
            pos=(BUTTON_X, BUTTON_TOP - row * BUTTON_STEP),
        )
        toggles[point_type] = toggle
    return toggles


def _add_button(
    plotter: vedo.Plotter,
    function: Callable[..., None],
    states: Sequence[str],
    pos: tuple[float, float],
) -> vedo.addons.Button:
    """Add a two-state button to the plotter's current renderer.

    Does what ``Plotter.add_button`` does, except that the 2D actor is added
    with ``AddActor``: VTK 9.7 removed ``vtkRenderer.AddActor2D``, which
    vedo 2026.6 still calls. Works offscreen too (no interactor needed).

    Returns:
        The button; its pick event calls ``function``.
    """
    button = vedo.addons.Button(
        function,
        states=list(states),
        c=("white", "white"),
        bc=("gray4", "gray2"),
        pos=pos,
        size=14,
        font="Calco",
        bold=False,
    )
    # Buttons are centered on their position by default; anchor the left
    # edge instead so a column of labels of different lengths lines up.
    button.pos(pos, "top-left")
    plotter.renderer.AddActor(button.actor)
    button.function_id = button.actor.AddObserver("PickEvent", button.function)
    plotter.buttons.append(button)
    return button


def _object_actors(
    channels: dict[str, LearnedObject],
    learned: dict[str, dict[str, dict[str, LearnedObject]]],
    label_colors: dict[str, str],
    color_by: str,
    point_size: int,
) -> dict[str, list]:
    """Build one object's vedo actors, grouped by point type.

    Returns:
        Per point type -- a sensor channel or a ``channel: child`` label --
        its actors.
    """
    actors: dict[str, list] = defaultdict(list)
    extent = np.ptp(
        np.vstack([c.points.locations for c in channels.values()]), axis=0
    ).max()
    labels = child_labels(channels, learned)
    for channel, obj in channels.items():
        xyz = obj.points.locations
        if channel in labels:
            node_labels = np.array(labels[channel])
            for label in sorted(set(node_labels)):
                spheres = vedo.Spheres(
                    xyz[node_labels == label], r=0.012 * extent, c=label_colors[label]
                )
                actors[label].append(spheres)
        else:
            points = vedo.Points(xyz, r=point_size)
            if color_by == STORED_COLOR:
                points.pointcolors = (obj.colors * 255).astype(np.uint8)
            else:
                values = obj.points[color_by]
                points.cmap("viridis", values[:, 0] if values.ndim == 2 else values)
                points.add_scalarbar(title=color_by)
            actors[channel].append(points)
    return dict(actors)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "model",
        nargs="?",
        help="model.pt, run directory, or run name under MODELS_DIR "
        "(default: the current chain's newest trained stage)",
    )
    parser.add_argument(
        "--lm",
        default=DEFAULT_LM,
        help=f"learning module to show (default: {DEFAULT_LM}, "
        "or the first with objects)",
    )
    parser.add_argument(
        "--objects", nargs="+", help="object ids to show (default: all)"
    )
    parser.add_argument(
        "--color-by",
        default=STORED_COLOR,
        help="'color' for the stored color, or a feature name (default: color)",
    )
    parser.add_argument("--point-size", type=int, default=6)
    parser.add_argument(
        "--screenshot", help="render offscreen to this file instead of a window"
    )
    args = parser.parse_args()
    inspect_model(
        args.model,
        lm=args.lm,
        objects=args.objects,
        color_by=args.color_by,
        point_size=args.point_size,
        screenshot=args.screenshot,
    )
