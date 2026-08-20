#!/usr/bin/env python
# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# ruff: noqa: DOC201,DOC501

"""Visualize learned 2D sensor object models.

The visualizer renders learned graph-memory points with optional edge tangent and
surface normal overlays. It is intentionally kept as one sandbox script, but the
data preparation, layers, controls, and CLI are separated so future controls can
be added without editing a large closure.
"""

from __future__ import annotations

import argparse
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch

if __package__:
    from .model_loading_utils import load_object_model
else:
    from model_loading_utils import load_object_model

if TYPE_CHECKING:
    from vedo import Plotter


DEFAULT_EDGE_STRENGTH_THRESHOLD = 0.1
DEFAULT_COHERENCE_THRESHOLD = 0.5
DEFAULT_UNSCALED_EDGE_SCALE = 0.002
DEFAULT_POINT_SIZE = 21
DEFAULT_TANGENT_LINE_WIDTH = 5
DEFAULT_NORMAL_LINE_WIDTH = 2
DEFAULT_NORMAL_SCALE = 0.01
DEFAULT_WINDOW_SIZE = (1400, 1000)
DEFAULT_BUTTON_POS = (0.55, 0.05)
BUTTON_X_SPACING = 0.13
FALLBACK_POINT_COLOR = np.array([128, 128, 128], dtype=np.uint8)


@dataclass(frozen=True)
class VisualizationConfig:
    """Runtime configuration for learned-model visualization."""

    edge_strength_threshold: float = DEFAULT_EDGE_STRENGTH_THRESHOLD
    coherence_threshold: float = DEFAULT_COHERENCE_THRESHOLD
    unscaled_edge_scale: float = DEFAULT_UNSCALED_EDGE_SCALE
    point_size: int = DEFAULT_POINT_SIZE
    tangent_line_width: int = DEFAULT_TANGENT_LINE_WIDTH
    normal_line_width: int = DEFAULT_NORMAL_LINE_WIDTH
    normal_scale: float = DEFAULT_NORMAL_SCALE
    show_normals: bool = False
    window_size: tuple[int, int] = DEFAULT_WINDOW_SIZE


@dataclass(frozen=True)
class PreparedModelView:
    """Validated arrays used by the Vedo visualizer."""

    points: np.ndarray
    colors: np.ndarray
    normals: np.ndarray | None
    tangents: np.ndarray | None
    world_edge_tangents: np.ndarray | None
    edge_mask: np.ndarray | None
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    center: np.ndarray

    @property
    def has_edges(self) -> bool:
        """Whether the view has enough data to render edge tangent overlays."""
        return (
            self.has_local_edge_tangents or self.has_world_edge_tangents
        ) and self.edge_mask is not None

    @property
    def has_local_edge_tangents(self) -> bool:
        """Whether local 2D pose-vector tangents are available."""
        return self.tangents is not None

    @property
    def has_world_edge_tangents(self) -> bool:
        """Whether world-coordinate edge tangents are available."""
        return self.world_edge_tangents is not None

    @property
    def has_normals(self) -> bool:
        """Whether the view has enough data to render surface normal overlays."""
        return self.normals is not None


@dataclass(frozen=True)
class ControlSpec:
    """Declarative button definition for a visualizer control."""

    label_off: str
    label_on: str
    callback_name: str
    enabled: Callable[[LearnedModelVisualizer], bool]
    active: Callable[[LearnedModelVisualizer], bool]


def normalize_rows_safe(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize row vectors while leaving near-zero rows finite.

    This is deliberately separate from
    ``tbp.monty.frameworks.utils.spatial_arithmetics.normalize``, which is a
    single-vector helper that raises on near-zero inputs. Visualization data can
    contain invalid or zero rows, so this helper keeps array shape stable and the
    caller filters invalid rows afterward.
    """
    rows = np.asarray(values, dtype=float)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return rows / norms


def _as_feature_array(value) -> np.ndarray:
    """Convert tensors or array-like feature values to numpy arrays."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _extract_pose_vectors(
    features: dict,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract surface normals and edge tangents from ``pose_vectors``."""
    if "pose_vectors" not in features:
        return None, None

    pose_vectors = np.asarray(features["pose_vectors"], dtype=float)
    if pose_vectors.ndim == 2 and pose_vectors.shape[1] == 9:
        pose_vectors = pose_vectors.reshape(-1, 3, 3)
    if pose_vectors.ndim != 3 or pose_vectors.shape[1:] != (3, 3):
        print(
            "[viz] Warning: expected pose_vectors with shape (N, 9) or "
            f"(N, 3, 3), got {pose_vectors.shape}; skipping vector overlays."
        )
        return None, None

    return pose_vectors[:, 0, :], pose_vectors[:, 1, :]


def _extract_world_edge_tangents(features: dict) -> np.ndarray | None:
    """Extract normalized world-space edge tangents when present."""
    if "world_edge_tangent" not in features:
        return None

    world_tangents = np.asarray(features["world_edge_tangent"], dtype=float)
    if world_tangents.ndim != 2 or world_tangents.shape[1] != 3:
        print(
            "[viz] Warning: expected world_edge_tangent with shape (N, 3), "
            f"got {world_tangents.shape}; skipping world edge overlays."
        )
        return None

    return normalize_rows_safe(world_tangents)


def _hsv_to_rgb_uint8(hsv: np.ndarray) -> np.ndarray:
    """Convert HSV values in [0, 1] to uint8 RGB colors."""
    hsv = np.asarray(hsv, dtype=float)
    h = np.mod(hsv[:, 0], 1.0)
    s = np.clip(hsv[:, 1], 0.0, 1.0)
    v = np.clip(hsv[:, 2], 0.0, 1.0)

    sector = np.floor(h * 6.0).astype(int)
    f = h * 6.0 - sector
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    rgb = np.zeros((len(hsv), 3), dtype=float)
    sector_mod = sector % 6
    choices = [
        (v, t, p),
        (q, v, p),
        (p, v, t),
        (p, q, v),
        (t, p, v),
        (v, p, q),
    ]
    for idx, channels in enumerate(choices):
        mask = sector_mod == idx
        rgb[mask] = np.column_stack([channel[mask] for channel in channels])

    return np.rint(rgb * 255.0).astype(np.uint8)


def _compute_point_colors(features: dict, n_points: int) -> np.ndarray:
    """Return per-point RGB colors, preferring HSV then RGBA then gray."""
    colors = np.tile(FALLBACK_POINT_COLOR, (n_points, 1))
    if "hsv" in features:
        hsv = np.asarray(features["hsv"])
        if hsv.ndim == 2 and hsv.shape[1] >= 3:
            n = min(n_points, hsv.shape[0])
            colors[:n] = _hsv_to_rgb_uint8(hsv[:n, :3])
            return colors
        print(f"[viz] Warning: invalid hsv shape {hsv.shape}; falling back to rgba.")

    if "rgba" not in features:
        return colors

    rgba = np.asarray(features["rgba"])
    if rgba.ndim != 2 or rgba.shape[1] < 3:
        print(f"[viz] Warning: invalid rgba shape {rgba.shape}; using gray points.")
        return colors

    n = min(n_points, rgba.shape[0])
    colors[:n] = np.clip(rgba[:n, :3], 0, 255).astype(np.uint8)
    return colors


def _compute_edge_mask(
    features: dict,
    n_points: int,
    config: VisualizationConfig,
) -> np.ndarray | None:
    """Compute edge mask from edge features."""
    if "edge_strength" not in features or "coherence" not in features:
        print(
            "[viz] Warning: edge_strength/coherence not found in features. "
            "Edge overlays and edge-only filtering will be unavailable."
        )
        return None

    edge_strength = np.asarray(features["edge_strength"], dtype=float).reshape(-1)
    coherence = np.asarray(features["coherence"], dtype=float).reshape(-1)
    n = min(n_points, len(edge_strength), len(coherence))

    edge_mask = np.zeros(n_points, dtype=bool)
    edge_mask[:n] = (edge_strength[:n] > config.edge_strength_threshold) & (
        coherence[:n] > config.coherence_threshold
    )

    print(
        "[viz] Edge mask "
        f"(edge_strength>{config.edge_strength_threshold:g} & "
        f"coherence>{config.coherence_threshold:g}): "
        f"{int(edge_mask.sum())}/{n_points} edge points"
    )
    return edge_mask


def prepare_model_view(
    model_data: dict,
    config: VisualizationConfig | None = None,
) -> PreparedModelView:
    """Convert loaded model data into validated arrays for visualization."""
    config = config or VisualizationConfig()
    points = np.asarray(model_data["points"], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}")
    if len(points) == 0:
        raise ValueError("Cannot visualize an empty point cloud")

    features = {
        key: _as_feature_array(value)
        for key, value in model_data.get("features", {}).items()
    }

    print(f"[viz] Available features: {list(features.keys())}")
    print(f"[viz] Points shape: {points.shape}")

    colors = _compute_point_colors(features, len(points))
    normals, tangents = _extract_pose_vectors(features)
    world_edge_tangents = _extract_world_edge_tangents(features)
    edge_mask = _compute_edge_mask(features, len(points), config)

    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    center = points.mean(axis=0)
    print("[viz] Point bounds:")
    print(f"  X: [{bounds_min[0]:.3f}, {bounds_max[0]:.3f}]")
    print(f"  Y: [{bounds_min[1]:.3f}, {bounds_max[1]:.3f}]")
    print(f"  Z: [{bounds_min[2]:.3f}, {bounds_max[2]:.3f}]")
    print(f"  Center: {center}")

    return PreparedModelView(
        points=points,
        colors=colors,
        normals=normals,
        tangents=tangents,
        world_edge_tangents=world_edge_tangents,
        edge_mask=edge_mask,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        center=center,
    )


def compute_camera(view: PreparedModelView) -> dict:
    """Compute a camera looking down at the point cloud from +Z."""
    ranges = view.bounds_max - view.bounds_min
    max_range = max(float(ranges.max()), 1e-6)
    camera_distance = max_range * 1.5
    camera_pos = (
        float(view.center[0]),
        float(view.center[1]),
        float(view.center[2] + camera_distance),
    )
    return {
        "pos": camera_pos,
        "focal_point": view.center,
        "view_angle": 45,
    }


class Layer:
    """Base class for reusable Vedo actor layers."""

    def __init__(self, visible: bool = True) -> None:
        self.visible = visible
        self.actors = []

    def clear(self, plotter: Plotter) -> None:
        """Remove all actors owned by this layer."""
        if self.actors:
            plotter.remove(self.actors)
            self.actors.clear()

    def redraw(
        self,
        plotter: Plotter,
        view: PreparedModelView,
        config: VisualizationConfig,
        indices: np.ndarray,
    ) -> None:
        """Redraw this layer for active point indices."""
        self.clear(plotter)
        if not self.visible:
            return
        self.actors = self.build(view, config, indices)
        if self.actors:
            plotter.add(*self.actors)

    def build(
        self,
        view: PreparedModelView,
        config: VisualizationConfig,
        indices: np.ndarray,
    ) -> list:
        """Build Vedo actors for this layer."""
        raise NotImplementedError


class PointCloudLayer(Layer):
    """Point cloud layer colored by rgba feature values."""

    def build(
        self,
        view: PreparedModelView,
        config: VisualizationConfig,
        indices: np.ndarray,
    ) -> list:
        from vedo import Points  # noqa: PLC0415

        cloud = Points(view.points[indices], r=config.point_size)
        cloud.pointcolors = view.colors[indices].tolist()
        return [cloud]


class VectorLineLayer(Layer):
    """Line glyph layer for tangents or normals."""

    def __init__(
        self,
        vector_name: str,
        color,
        line_width_getter: Callable[[VisualizationConfig], int],
        scale_getter: Callable[[PreparedModelView, VisualizationConfig], np.ndarray],
        mask_getter: Callable[[PreparedModelView], np.ndarray | None],
        vector_getter: Callable[[PreparedModelView], np.ndarray | None] | None = None,
        visible: bool = True,
    ) -> None:
        super().__init__(visible=visible)
        self.vector_name = vector_name
        self.color = color
        self.line_width_getter = line_width_getter
        self.scale_getter = scale_getter
        self.mask_getter = mask_getter
        self.vector_getter = vector_getter

    def _vectors_for_view(self, view: PreparedModelView) -> np.ndarray | None:
        if self.vector_getter is not None:
            return self.vector_getter(view)
        if self.vector_name == "tangents":
            return view.tangents
        if self.vector_name == "normals":
            return view.normals
        raise ValueError(f"Unknown vector field: {self.vector_name}")

    def build(
        self,
        view: PreparedModelView,
        config: VisualizationConfig,
        indices: np.ndarray,
    ) -> list:
        from vedo import Line  # noqa: PLC0415

        vectors = self._vectors_for_view(view)
        if vectors is None:
            return []

        n_common = min(len(view.points), len(vectors))
        in_bounds = indices[indices < n_common]
        if len(in_bounds) == 0:
            return []

        mask = self.mask_getter(view)
        if mask is not None:
            in_bounds = in_bounds[mask[in_bounds]]
        if len(in_bounds) == 0:
            return []

        normalized = normalize_rows_safe(vectors[:n_common])
        valid = np.isfinite(normalized[in_bounds]).all(axis=1) & (
            np.linalg.norm(vectors[in_bounds], axis=1) > 1e-9
        )
        line_indices = in_bounds[valid]
        if len(line_indices) == 0:
            return []

        scales = self.scale_getter(view, config)
        line_width = self.line_width_getter(config)
        lines = []
        for idx in line_indices:
            half = scales[idx] / 2.0
            start = view.points[idx] - half * normalized[idx]
            end = view.points[idx] + half * normalized[idx]
            lines.append(Line(start, end, c=self.color, lw=line_width))
        return lines


class LearnedModelVisualizer:
    """Interactive Vedo visualizer for prepared learned model data."""

    def __init__(
        self,
        view: PreparedModelView,
        config: VisualizationConfig,
        title: str | None = None,
    ) -> None:
        from vedo import Plotter  # noqa: PLC0415

        self.view = view
        self.config = config
        self.title = title or "Learned Point Cloud"
        self.plotter = Plotter(size=config.window_size, title=self.title)
        self.initial_camera = compute_camera(self.view)
        self.show_edge_only = False
        self._set_default_edge_vector_space()
        self.layers = {
            "points": PointCloudLayer(visible=True),
            "unscaled_edges": VectorLineLayer(
                vector_name="edge_tangents",
                color=(200, 0, 0),
                line_width_getter=lambda cfg: cfg.tangent_line_width,
                scale_getter=self._unscaled_edge_lengths,
                mask_getter=lambda view: view.edge_mask,
                vector_getter=self._edge_vectors_for_view,
                visible=view.has_edges,
            ),
            "normals": VectorLineLayer(
                vector_name="normals",
                color="blue",
                line_width_getter=lambda cfg: cfg.normal_line_width,
                scale_getter=self._normal_lengths,
                mask_getter=lambda _view: None,
                visible=config.show_normals and view.has_normals,
            ),
        }

    def _active_indices(self) -> np.ndarray:
        if self.show_edge_only and self.view.edge_mask is not None:
            edge_indices = np.where(self.view.edge_mask)[0]
            if len(edge_indices) == 0:
                print("[viz] Warning: no edge points found; showing all points.")
                return np.arange(len(self.view.points))
            return edge_indices
        return np.arange(len(self.view.points))

    def _set_default_edge_vector_space(self) -> None:
        """Prefer world tangents when checkpoints include them."""
        if self.view.has_world_edge_tangents:
            self.edge_vector_space = "world"
        else:
            self.edge_vector_space = "local"

    def _edge_vectors_for_view(self, view: PreparedModelView) -> np.ndarray | None:
        """Return the active edge tangent vector source with graceful fallback."""
        if self.edge_vector_space == "world":
            if view.world_edge_tangents is not None:
                return view.world_edge_tangents
            return view.tangents
        if view.tangents is not None:
            return view.tangents
        return view.world_edge_tangents

    def _unscaled_edge_lengths(
        self, view: PreparedModelView, config: VisualizationConfig
    ) -> np.ndarray:
        return np.full(len(view.points), config.unscaled_edge_scale, dtype=float)

    def _normal_lengths(
        self, view: PreparedModelView, config: VisualizationConfig
    ) -> np.ndarray:
        return np.full(len(view.points), config.normal_scale, dtype=float)

    def redraw(self) -> None:
        """Redraw all registered layers."""
        indices = self._active_indices()
        for layer in self.layers.values():
            layer.redraw(self.plotter, self.view, self.config, indices)
        self.plotter.render()

    def _toggle_edge_filter(self, button, _event) -> None:
        self.show_edge_only = not self.show_edge_only
        button.switch()
        if self.show_edge_only:
            n_edges = (
                int(self.view.edge_mask.sum()) if self.view.edge_mask is not None else 0
            )
            print(f"[viz] Showing {n_edges} edge points")
        else:
            print(f"[viz] Showing all {len(self.view.points)} points")
        self.redraw()

    def _toggle_layer(self, layer_name: str, button, _event) -> None:
        layer = self.layers[layer_name]
        layer.visible = not layer.visible
        button.switch()
        self.redraw()

    def _toggle_edge_vector_space(self, button, _event) -> None:
        self.edge_vector_space = (
            "local" if self.edge_vector_space == "world" else "world"
        )
        button.switch()
        print(f"[viz] Showing {self.edge_vector_space} edge tangents")
        self.redraw()

    def _reset_camera(self, _button=None, _event=None) -> None:
        """Restore the initial screenshot-friendly camera view."""
        camera = getattr(self.plotter, "camera", None)
        if camera is not None:
            camera.SetPosition(*self.initial_camera["pos"])
            camera.SetFocalPoint(*self.initial_camera["focal_point"])
            camera.SetViewUp(0, 1, 0)
            camera.SetViewAngle(self.initial_camera["view_angle"])
            reset_clipping = getattr(self.plotter, "reset_camera_clipping_range", None)
            if reset_clipping is not None:
                reset_clipping()
            elif getattr(self.plotter, "renderer", None) is not None:
                self.plotter.renderer.ResetCameraClippingRange()
        self.plotter.render()

    def _control_specs(self) -> list[ControlSpec]:
        return [
            ControlSpec(
                label_off=" All Points ",
                label_on=" Edge Only ",
                callback_name="_toggle_edge_filter",
                enabled=lambda viz: viz.view.edge_mask is not None,
                active=lambda viz: viz.show_edge_only,
            ),
            ControlSpec(
                label_off=" Edges Off ",
                label_on=" Edges On ",
                callback_name="unscaled_edges",
                enabled=lambda viz: viz.view.has_edges,
                active=lambda viz: viz.layers["unscaled_edges"].visible,
            ),
            ControlSpec(
                label_off=" World Edges ",
                label_on=" Local Edges ",
                callback_name="edge_vector_space",
                enabled=lambda viz: (
                    viz.view.has_local_edge_tangents
                    and viz.view.has_world_edge_tangents
                ),
                active=lambda viz: viz.edge_vector_space == "local",
            ),
            ControlSpec(
                label_off=" Normals Off ",
                label_on=" Normals On ",
                callback_name="normals",
                enabled=lambda viz: viz.view.has_normals,
                active=lambda viz: viz.layers["normals"].visible,
            ),
            ControlSpec(
                label_off=" Reset Camera ",
                label_on=" Reset Camera ",
                callback_name="_reset_camera",
                enabled=lambda _viz: True,
                active=lambda _viz: False,
            ),
        ]

    def _control_position(self, button_index: int) -> tuple[float, float]:
        """Return the normalized position for a horizontal control row."""
        x, y = DEFAULT_BUTTON_POS
        return x + button_index * BUTTON_X_SPACING, y

    def add_controls(self) -> None:
        """Create all enabled buttons from the declarative control registry."""
        button_index = 0
        for spec in self._control_specs():
            if not spec.enabled(self):
                continue

            pos = self._control_position(button_index)
            states = (
                [spec.label_on, spec.label_off]
                if spec.active(self)
                else [spec.label_off, spec.label_on]
            )
            if spec.callback_name == "_toggle_edge_filter":
                callback = self._toggle_edge_filter
            elif spec.callback_name == "_reset_camera":
                callback = self._reset_camera
            elif spec.callback_name == "edge_vector_space":
                callback = self._toggle_edge_vector_space
            else:

                def callback(button, event, layer_name=spec.callback_name):
                    self._toggle_layer(layer_name, button, event)

            self.plotter.add_button(
                callback,
                pos=pos,
                states=states,
                size=20,
                font="Calco",
            )
            button_index += 1

    def show(self) -> None:
        """Render the visualizer and enter Vedo's interactive loop."""
        self.redraw()
        self.add_controls()
        self.plotter.show(
            axes=dict(xtitle="X", ytitle="Y", ztitle="Z"),
            viewup="y",
            camera=self.initial_camera,
            interactive=True,
        )


def visualize_point_cloud_interactive(
    model_data: dict,
    title: str | None = None,
) -> None:
    """Create interactive 3D visualization with Vedo."""
    config = VisualizationConfig()
    view = prepare_model_view(model_data, config)
    LearnedModelVisualizer(view, config, title=title).show()


def _load_available_objects(model_path: Path, lm_id: int) -> list[str]:
    """Load checkpoint metadata and return graph-memory object names."""
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    graph_memory = state_dict["lm_dict"][lm_id]["graph_memory"]
    return list(graph_memory.keys())


def _select_objects(
    available_objects: list[str], requested_objects: list[str] | None
) -> list[str]:
    """Validate and return requested object names."""
    if requested_objects is None:
        return available_objects

    missing = [name for name in requested_objects if name not in available_objects]
    if missing:
        raise ValueError(
            f"Requested objects not found: {missing}. "
            f"Available objects: {available_objects}"
        )
    return requested_objects


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize learned 2D sensor models.",
        epilog=(
            "Examples:\n"
            "  conda activate tbp.monty\n"
            '  export PYTHONPATH="$(pwd)/src"\n'
            "  python sandbox/visualize_learned_model.py --model-path "
            "path/to/model.pt\n"
            "  python sandbox/visualize_learned_model.py --model-path "
            "path/to/model.pt --objects disk cylinder"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the pretrained model.pt checkpoint.",
    )
    parser.add_argument("--lm", type=int, default=0, help="Learning module index.")
    parser.add_argument(
        "--objects",
        type=str,
        nargs="+",
        default=None,
        help="Object names to visualize. Defaults to all objects in graph memory.",
    )
    return parser.parse_args()


def main() -> None:
    """Load selected object models and visualize them one at a time."""
    args = parse_args()
    model_path = args.model_path.expanduser()
    config = VisualizationConfig()

    print("Loading model metadata...")
    available_objects = _load_available_objects(model_path, args.lm)
    print(f"\nAvailable objects: {available_objects}")

    try:
        selected_objects = _select_objects(available_objects, args.objects)
    except ValueError as e:
        raise SystemExit(str(e)) from e

    for object_name in selected_objects:
        print(f"\nProcessing {object_name}...")
        try:
            model_data = load_object_model(model_path, object_name, lm_id=args.lm)
            print(f"  Points shape: {model_data['points'].shape}")
            print(f"  Available features: {list(model_data['features'].keys())}")

            view = prepare_model_view(model_data, config)
            visualizer = LearnedModelVisualizer(
                view,
                config,
                title=f"Learned Point Cloud: {object_name}",
            )
            visualizer.show()
        except Exception as e:  # noqa: BLE001
            print(f"  Error processing {object_name}: {e}")
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
