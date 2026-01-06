"""Script to visualize the learned model using 2D Sensor Module."""

from pathlib import Path
import numpy as np
import torch
from vedo import Plotter, Points, Line

from model_loading_utils import load_object_model


def _normalize_rows(V, eps=1e-12):
    V = np.asarray(V, float)
    n = np.linalg.norm(V, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return V / n


def visualize_point_cloud_interactive(
    model_data,
    title=None,
    arrow_scale=0.01,
    *,
    tangent_color="black",
    tangent_lw=3,
    show_unscaled_edge_lines=True,
):
    """Create interactive 3D visualization with Vedo.

    Args:
        model_data: dict with keys:
            - points: (N,3) world coords
            - features: dict. Expected options:
                * 'pose_vectors': (N,3,3) with [normal, edge_tangent, edge_perp] per point
                * 'pose_from_edge': (N,) boolean indicating if pose comes from edge
                * optional 'rgba': (N,3 or 4)
                * optional 'pose_fully_defined': (N,) binary or NaN indicating if pose is fully defined
        title: window title.
        arrow_scale: length (in world units) of each tangent line.
        tangent_color: Vedo color for tangent lines.
        tangent_lw: line width for tangent lines.
        show_unscaled_edge_lines: If True, draw red unscaled edge lines. Defaults to False.
    """
    points = np.asarray(model_data["points"], float)
    features = model_data["features"]

    # Debug: print available features
    print(f"[viz] Available features: {list(features.keys())}")

    # State variables for edge lines
    edge_lines = []
    unscaled_edge_lines = []

    # State variables for color mode
    # Build list of available color modes
    available_color_modes = []
    mode_names = []
    if "rgba" in features:
        available_color_modes.append("rgba")
        mode_names.append(" RGBA ")
    if "pose_fully_defined" in features:
        available_color_modes.append("pose_fully_defined")
        mode_names.append(" Pose Defined ")

    # Debug: print available color modes
    print(f"[viz] Available color modes: {available_color_modes}")
    print(f"[viz] Mode names: {mode_names}")

    # Default to first available mode
    color_mode_idx = 0
    point_cloud_obj = None

    def _get_colors_for_mode(mode_name, n_points):
        """Get colors for the specified mode."""
        if mode_name == "rgba" and "rgba" in features:
            return features["rgba"][:, :3].astype(np.uint8).tolist()
        if mode_name == "pose_fully_defined" and "pose_fully_defined" in features:
            pfd = np.asarray(features["pose_fully_defined"], float).reshape(-1)
            n_pfd = len(pfd)
            n_common = min(n_pfd, n_points)

            colors = np.zeros((n_points, 3), dtype=np.uint8)
            # Default all to gray (for any points beyond pfd length)
            colors.fill(128)

            # Map: 1 -> green, 0 -> red, NaN -> gray
            for i in range(n_common):
                val = pfd[i]
                if np.isnan(val):
                    colors[i] = [128, 128, 128]  # gray
                elif np.isclose(val, 1.0, atol=1e-6):
                    colors[i] = [0, 255, 0]  # green
                elif np.isclose(val, 0.0, atol=1e-6):
                    colors[i] = [255, 0, 0]  # red
                else:
                    colors[i] = [128, 128, 128]  # gray (default for unexpected values)

            return colors.tolist()
        return None

    def toggle_color_mode_callback(button, _event):
        """Cycle through available color modes."""
        nonlocal color_mode_idx, point_cloud_obj

        if available_color_modes:
            color_mode_idx = (color_mode_idx + 1) % len(available_color_modes)
            button.switch()

            if point_cloud_obj is not None:
                n_points = len(points)
                colors = _get_colors_for_mode(
                    available_color_modes[color_mode_idx], n_points
                )
                if colors is not None:
                    point_cloud_obj.pointcolors = colors
                    plotter.render()

    plotter = Plotter(size=(1400, 1000), title=title or "Learned Point Cloud")

    # ----- Point colors -----
    n_points = len(points)
    if available_color_modes:
        # Use the default (first) color mode
        colors = _get_colors_for_mode(available_color_modes[color_mode_idx], n_points)
        point_cloud_obj = Points(points, r=10)
        if colors is not None:
            point_cloud_obj.pointcolors = colors
        plotter.add(point_cloud_obj)
    else:
        point_cloud_obj = Points(points, r=10).color("gray")
        plotter.add(point_cloud_obj)

    # ----- Collect 3D edge tangents (WORLD frame) -----
    tangents = None
    if "pose_vectors" in features:
        # Expect shape (N, 9)
        pv = np.asarray(features["pose_vectors"], float)
        if pv.ndim == 2 and pv.shape[1] == 9:
            pv = pv.reshape(-1, 3, 3)
        tangents = pv[:, 1, :]

    # Require pose_from_edge to draw edges
    if "pose_from_edge" not in features:
        print(
            "[viz] Warning: pose_from_edge not found in features. Edge lines will not be drawn."
        )
        edge_mask = None
    else:
        edge_mask = np.asarray(features["pose_from_edge"], bool).reshape(-1)

    # ----- Draw tangents as Lines -----
    def draw_edges():
        """Draw edge tangent lines for points where pose_from_edge is True."""
        nonlocal edge_lines, unscaled_edge_lines

        # Remove existing edge lines
        if edge_lines:
            plotter.remove(edge_lines)
            edge_lines.clear()
        if unscaled_edge_lines:
            plotter.remove(unscaled_edge_lines)
            unscaled_edge_lines.clear()

        if tangents is None or edge_mask is None:
            return

        points_arr = np.asarray(points, float)
        tangents_arr = np.asarray(tangents, float)

        # Align lengths defensively (some rows may be missing/extra)
        n_points = points_arr.shape[0]
        n_tangents = tangents_arr.shape[0]
        n_mask = edge_mask.shape[0]
        n_common = min(n_points, n_tangents, n_mask)

        if n_common == 0:
            print("[viz] No common rows to draw tangents.")
            return

        P = points_arr[:n_common]
        T = tangents_arr[:n_common]
        EM = edge_mask[:n_common]

        # Extract coherence and edge_strength if available
        scale_factors = None
        if "coherence" in features and "edge_strength" in features:
            coherence_arr = np.asarray(features["coherence"], float).reshape(-1)
            edge_strength_arr = np.asarray(features["edge_strength"], float).reshape(-1)
            n_coherence = coherence_arr.shape[0]
            n_edge_strength = edge_strength_arr.shape[0]
            n_common_scale = min(n_common, n_coherence, n_edge_strength)
            if n_common_scale > 0:
                coherence_vals = coherence_arr[:n_common_scale]
                edge_strength_vals = edge_strength_arr[:n_common_scale]
                # Compute (coherence * edge_strength) / 4, which ranges from [0, 1]
                # since coherence * edge_strength ranges from [0, 4]
                scale_factors = (coherence_vals * edge_strength_vals) / 4.0
                # Clamp to [0, 1] to ensure valid scaling
                scale_factors = np.clip(scale_factors, 0.0, 1.0)
                print(
                    f"[viz] Using (coherence * edge_strength) / 4 to scale edge line lengths (range: [{scale_factors.min():.3f}, {scale_factors.max():.3f}])"
                )
            else:
                print(
                    "[viz] Warning: coherence/edge_strength found but have no valid values"
                )
        else:
            if "coherence" not in features:
                print(
                    "[viz] No coherence feature found; using fixed scale for edge lines"
                )
            if "edge_strength" not in features:
                print(
                    "[viz] No edge_strength feature found; using fixed scale for edge lines"
                )

        # Print counts for debugging (only on first draw)
        if len(edge_lines) == 0:
            n_true = int(EM.sum())
            n_false = int((~EM).sum())
            print(
                f"[viz] pose_from_edge: True={n_true}, False={n_false}, Total={n_common}"
            )

        # Normalize & validate
        T = _normalize_rows(T)
        valid = np.isfinite(T).all(axis=1) & (np.linalg.norm(T, axis=1) > 1e-9)

        # Keep only edge-derived & valid
        keep = EM & valid
        idx = np.where(keep)[0]

        # Build lines scaled by (coherence * edge_strength) / 4
        new_lines = []
        # Also build unscaled lines (using fixed scale of 1.0)
        new_unscaled_lines = []
        for i in idx:
            p0 = P[i]
            # Scale by (coherence * edge_strength) / 4 if available, otherwise use fixed scale
            if scale_factors is not None and i < len(scale_factors):
                scale_factor = scale_factors[i]
            else:
                scale_factor = 1.0
            # Center the line around the point
            # arrow_scale (0.002) is the maximum length
            half_scale = (arrow_scale * scale_factor) / 2
            p_start = p0 - half_scale * T[i]
            p_end = p0 + half_scale * T[i]
            new_lines.append(Line(p_start, p_end, c=tangent_color, lw=tangent_lw))

            # Also draw unscaled edge (always using full arrow_scale) if enabled
            if show_unscaled_edge_lines:
                half_scale_unscaled = 0.002 / 2
                p_start_unscaled = p0 - half_scale_unscaled * T[i]
                p_end_unscaled = p0 + half_scale_unscaled * T[i]
                # Use mid-red with transparency for unscaled edges
                new_unscaled_lines.append(
                    Line(
                        p_start_unscaled,
                        p_end_unscaled,
                        c=(200, 0, 0),
                        alpha=0.6,
                        lw=tangent_lw - 1,
                    )
                )

        if new_lines:
            plotter.add(*new_lines)
            edge_lines.extend(new_lines)
        if new_unscaled_lines:
            plotter.add(*new_unscaled_lines)
            unscaled_edge_lines.extend(new_unscaled_lines)
        if new_lines or new_unscaled_lines:
            plotter.render()
        else:
            print("[viz] No tangents to draw after filtering (mask/validity).")

    if tangents is not None and edge_mask is not None:
        draw_edges()
    else:
        print(
            "[viz] No pose_vectors/tangents or pose_from_edge found; skipping tangent rendering."
        )

    # ----- Add color mode toggle button -----
    button_y_pos = 0.05
    if available_color_modes:
        print(
            f"[viz] Creating color mode button with {len(mode_names)} states: {mode_names}"
        )
        plotter.add_button(
            toggle_color_mode_callback,
            pos=(0.85, button_y_pos),
            states=mode_names,
            size=20,
            font="Calco",
        )
        print("[viz] Color mode button created successfully")
    else:
        print("[viz] No color modes available, skipping color mode button")

    # ----- Axes & camera -----
    plotter.show(
        axes=dict(xtitle="X", ytitle="Y", ztitle="Z"),
        viewup="y",
        camera=dict(
            pos=(0, 1.5, 0.2),
            focal_point=(0, 1.5, 0),
            view_angle=45,
        ),
        interactive=True,
    )


if __name__ == "__main__":
    # Set up paths
    pretrained_model_path = Path(
        "~/tbp/results/monty/pretrained_models/2d_sensor/"
        "013_cylinder_tbp_vert_learning_2d_zoom30_blur5.0/pretrained/model.pt"
    ).expanduser()

    # Load the model to explore available objects
    print("Loading model...")
    state_dict = torch.load(pretrained_model_path, map_location="cpu")

    # Print available objects
    lm_id = 0
    graph_memory = state_dict["lm_dict"][lm_id]["graph_memory"]
    available_objects = list(graph_memory.keys())
    print(f"\nAvailable objects: {available_objects}")

    # Visualize each object interactively
    for object_name in available_objects:
        print(f"\nProcessing {object_name}...")

        try:
            # Load the model
            model_data = load_object_model(
                pretrained_model_path, object_name, lm_id=lm_id
            )

            print(f"  Points shape: {model_data['points'].shape}")
            print(f"  Available features: {list(model_data['features'].keys())}")

            # Create interactive visualization
            visualize_point_cloud_interactive(
                model_data,
                title=f"Learned Point Cloud: {object_name}",
            )

        except Exception as e:
            print(f"  Error processing {object_name}: {e}")
            import traceback

            traceback.print_exc()
            continue
