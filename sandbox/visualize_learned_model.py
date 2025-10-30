"""Script to visualize the learned model using 2D Sensor Module."""

from pathlib import Path
import numpy as np
import torch
import trimesh
from vedo import Plotter, Points, Line, Arrow, Mesh
from matplotlib.colors import hsv_to_rgb


def _normalize_rows(V, eps=1e-12):
    V = np.asarray(V, float)
    n = np.linalg.norm(V, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return V / n


def load_object_model(model_path, object_name, lm_id=0):
    """Load an object model from a pretraining experiment.

    Args:
        model_path: Path to the model checkpoint file.
        object_name: Name of the object to load.
        lm_id: ID of the learning module (default: 0).

    Returns:
        Dictionary containing:
            - points: (n_points, 3) array of 3D locations
            - features: dict mapping feature names to arrays
    """
    # Load the checkpoint
    state_dict = torch.load(model_path, map_location="cpu")

    # Navigate to the graph object
    graph_data = state_dict["lm_dict"][lm_id]["graph_memory"][object_name][
        "patch"
    ]._graph

    # Extract point positions
    pos = getattr(graph_data, "pos", None)
    if pos is None and hasattr(graph_data, "__dict__"):
        pos = graph_data.__dict__.get("pos")
    if pos is None:
        raise RuntimeError("Expected attribute 'pos' on patch object")

    if isinstance(pos, torch.Tensor):
        points = pos.detach().cpu().numpy().astype(float)
    else:
        points = np.asarray(pos, dtype=float)

    # Extract features
    feature_dict = {}
    feature_mapping = getattr(graph_data, "feature_mapping", {}) or {}
    x = getattr(graph_data, "x", None)
    if x is None and hasattr(graph_data, "__dict__"):
        x = graph_data.__dict__.get("x")

    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = np.asarray(x) if x is not None else None

    if x_np is not None and feature_mapping:
        for feature, idx in feature_mapping.items():
            # idx is expected to be [start, end)
            feature_data = np.asarray(x_np[:, idx[0] : idx[1]])
            feature_dict[feature] = feature_data

    return {
        "points": points,
        "features": feature_dict,
    }


def load_mesh(object_name, mesh_base_path=None):
    """Load a mesh file for the given object.

    Args:
        object_name: Name of the object to load.
        mesh_base_path: Base path to the meshes directory. If None, uses default path.

    Returns:
        trimesh.Trimesh or trimesh.Scene object, or None if file not found.
    """
    if mesh_base_path is None:
        mesh_base_path = Path("/Users/hlee/tbp/data/compositional_objects/meshes")
    else:
        mesh_base_path = Path(mesh_base_path)

    mesh_path = mesh_base_path / object_name / "textured.glb"

    try:
        with open(mesh_path, "rb") as f:
            mesh = trimesh.load_mesh(f, file_type="glb")
        return mesh
    except FileNotFoundError:
        print(f"  Mesh file not found: {mesh_path}")
        return None
    except Exception as e:
        print(f"  Error loading mesh: {e}")
        return None


def visualize_point_cloud_interactive(
    model_data,
    title=None,
    arrow_scale=0.002,
    mesh=None,
    mesh_translation=None,
    *,
    tangent_color="black",
    tangent_lw=3,
    max_tangent_lines=8000,  # cap for perf
    subsample_every=None,  # e.g. 5 to draw every 5th tangent
    prefer_edgey_points=True,  # if edge_strength available, draw the edgiest first
):
    """Create interactive 3D visualization with Vedo.

    Args:
        model_data: dict with keys:
            - points: (N,3) world coords
            - features: dict. Expected options:
                * 'pose_vectors': (N,3,3) with [normal, edge_tangent, edge_perp] per point
                  OR
                * 'edge_tangent': (N,3) directly
                * optional 'edge_strength': (N,) to prioritize which lines to draw
                * optional 'rgba': (N,3 or 4)
        title: window title.
        arrow_scale: length (in world units) of each tangent line.
        mesh: optional trimesh.Trimesh or trimesh.Scene to render alongside.
        mesh_translation: [x,y,z] offset for the mesh (default [0,1.5,0]).
        tangent_color: Vedo color for tangent lines.
        tangent_lw: line width for tangent lines.
        max_tangent_lines: performance guard; limit number of lines.
        subsample_every: if set, take every k-th tangent.
        prefer_edgey_points: if True and 'edge_strength' exists, prioritize those.
    """
    points = np.asarray(model_data["points"], float)
    features = model_data["features"]

    # State variables for surface normal arrows
    surface_normals_visible = False
    surface_normal_arrows = []

    # State variables for mesh visibility
    mesh_visible = False
    mesh_objects = []

    # State variables for color mode
    # 0 = Hue (full saturation), 1 = HSV->RGB (actual values), 2 = RGBA
    color_mode = 0 if "hsv" in features else 2
    point_cloud_obj = None

    def toggle_color_mode_callback(button, _event):
        """Toggle between hue, HSV->RGB, and RGBA coloring."""
        nonlocal color_mode, point_cloud_obj

        # Cycle through available modes
        if "hsv" in features and "rgba" in features:
            color_mode = (color_mode + 1) % 3
        elif "hsv" in features:
            color_mode = (color_mode + 1) % 2
        elif "rgba" in features:
            color_mode = 2

        button.switch()

        if point_cloud_obj is not None:
            if color_mode == 0 and "hsv" in features:
                # Mode 0: Hue only (full saturation and value)
                hsv_data = features["hsv"]
                hue = hsv_data[:, 0]
                n_points = len(hue)
                hsv_for_display = np.zeros((n_points, 3))
                hsv_for_display[:, 0] = hue
                hsv_for_display[:, 1] = 1.0
                hsv_for_display[:, 2] = 1.0
                rgb_colors = hsv_to_rgb(hsv_for_display)
                colors = (rgb_colors * 255).astype(np.uint8).tolist()
                point_cloud_obj.pointcolors = colors
            elif color_mode == 1 and "hsv" in features:
                # Mode 1: Full HSV to RGB conversion
                hsv_data = features["hsv"]
                rgb_colors = hsv_to_rgb(hsv_data)
                colors = (rgb_colors * 255).astype(np.uint8).tolist()
                point_cloud_obj.pointcolors = colors
            elif color_mode == 2 and "rgba" in features:
                # Mode 2: RGBA coloring
                colors = features["rgba"][:, :3].astype(np.uint8).tolist()
                point_cloud_obj.pointcolors = colors

        plotter.render()

    def toggle_surface_normals_callback(button, _event):
        """Toggle visibility of surface normal arrows."""
        nonlocal surface_normals_visible, surface_normal_arrows

        surface_normals_visible = not surface_normals_visible
        button.switch()

        if surface_normals_visible:
            _add_surface_normal_arrows(plotter, points, features, surface_normal_arrows)
        else:
            plotter.remove(surface_normal_arrows)
            surface_normal_arrows.clear()

        plotter.render()

    def toggle_mesh_callback(button, _event):
        """Toggle visibility of mesh."""
        nonlocal mesh_visible, mesh_objects

        mesh_visible = not mesh_visible
        button.switch()

        if mesh_visible:
            _add_mesh(plotter, mesh, mesh_translation, mesh_objects)
        else:
            plotter.remove(mesh_objects)
            mesh_objects.clear()

        plotter.render()

    plotter = Plotter(size=(1400, 1000), title=title or "Learned Point Cloud")

    # ----- Point colors -----
    if "hsv" in features:
        # Use hue component from HSV
        hsv_data = features["hsv"]
        hue = hsv_data[:, 0]  # Extract hue (first channel)

        # Convert hue to RGB colors for visualization
        # Create HSV array with full saturation and value to show pure hue colors
        n_points = len(hue)
        hsv_for_display = np.zeros((n_points, 3))
        hsv_for_display[:, 0] = hue  # Hue
        hsv_for_display[:, 1] = 1.0  # Full saturation
        hsv_for_display[:, 2] = 1.0  # Full value/brightness

        # Convert to RGB (0-255 range for vedo)
        rgb_colors = hsv_to_rgb(hsv_for_display)
        colors = (rgb_colors * 255).astype(np.uint8).tolist()

        point_cloud_obj = Points(points, r=10)
        point_cloud_obj.pointcolors = colors
        plotter.add(point_cloud_obj)
    elif "rgba" in features and features["rgba"].shape[1] >= 3:
        colors = features["rgba"][:, :3].astype(np.uint8).tolist()
        point_cloud_obj = Points(points, r=10)
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

    edge_mask = None
    if "pose_from_edge" in features:
        edge_mask = np.asarray(features["pose_from_edge"], bool).reshape(-1)

    # Optional: mask to points on object (if semantic ids provided)
    # If you have a mask, apply it here to 'points' and 'tangents'.

    # ----- Draw tangents as Lines -----
    if tangents is not None:
        points = np.asarray(points, float)
        tangents = np.asarray(tangents, float)

        # Align lengths defensively (some rows may be missing/extra)
        n_points = points.shape[0]
        n_tangents = tangents.shape[0]
        n_mask = edge_mask.shape[0] if edge_mask is not None else n_tangents
        n_common = min(n_points, n_tangents, n_mask)

        if n_common == 0:
            print("[viz] No common rows to draw tangents.")
        else:
            P = points[:n_common]
            T = tangents[:n_common]
            EM = (
                edge_mask[:n_common]
                if edge_mask is not None
                else np.ones((n_common,), dtype=bool)
            )

            # Print counts for debugging
            if edge_mask is not None:
                n_true = int(EM.sum())
                n_false = int((~EM).sum())
                print(
                    f"[viz] pose_from_edge: True={n_true}, False={n_false}, Total(masked)={n_common}"
                )

                # Normalize & validate
                T = _normalize_rows(T)
                valid = np.isfinite(T).all(axis=1) & (np.linalg.norm(T, axis=1) > 1e-9)

                # Keep only edge-derived & valid
                keep = EM & valid
                idx = np.where(keep)[0]

                # Build lines
                lines = []
                for i in idx:
                    p0 = P[i]
                    # Center the line around the point
                    half_scale = arrow_scale / 2
                    p_start = p0 - half_scale * T[i]
                    p_end = p0 + half_scale * T[i]
                    lines.append(Line(p_start, p_end, c=tangent_color, lw=tangent_lw))

                if lines:
                    plotter.add(*lines)
                else:
                    print("[viz] No tangents to draw after filtering (mask/validity).")
    else:
        print("[viz] No pose_vectors/tangents found; skipping tangent rendering.")

    # ----- Add surface normal toggle button -----
    surface_normals_button = plotter.add_button(
        toggle_surface_normals_callback,
        pos=(0.85, 0.05),
        states=[" Show Surface Normals ", " Hide Surface Normals "],
        size=20,
        font="Calco",
    )

    # ----- Add mesh toggle button (only if mesh is available) -----
    if mesh is not None:
        mesh_button = plotter.add_button(
            toggle_mesh_callback,
            pos=(0.85, 0.12),
            states=[" Show Mesh ", " Hide Mesh "],
            size=20,
            font="Calco",
        )

    # ----- Add color mode toggle button -----
    if "hsv" in features and "rgba" in features:
        color_mode_button = plotter.add_button(
            toggle_color_mode_callback,
            pos=(0.85, 0.19),
            states=[" Hue ", " HSV→RGB ", " RGBA "],
            size=20,
            font="Calco",
        )
    elif "hsv" in features:
        color_mode_button = plotter.add_button(
            toggle_color_mode_callback,
            pos=(0.85, 0.19),
            states=[" Hue ", " HSV→RGB "],
            size=20,
            font="Calco",
        )

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


def _add_surface_normal_arrows(plotter, points, features, surface_normal_arrows):
    """Add arrows showing surface normals for edge-detected points.

    Args:
        plotter: Vedo plotter instance
        points: (N,3) array of point locations
        features: dict containing pose_vectors and pose_from_edge
        surface_normal_arrows: list to store arrow objects for cleanup
    """
    arrow_length = 0.01

    # Check if we have the required features
    if "pose_vectors" not in features or "pose_from_edge" not in features:
        print(
            "[viz] No pose_vectors or pose_from_edge found; skipping surface normal rendering."
        )
        return

    # Get edge mask and pose vectors
    edge_mask = np.asarray(features["pose_from_edge"], bool).reshape(-1)
    pose_vectors = np.asarray(features["pose_vectors"], float)

    # Reshape pose_vectors if needed (should be N,3,3)
    if pose_vectors.ndim == 2 and pose_vectors.shape[1] == 9:
        pose_vectors = pose_vectors.reshape(-1, 3, 3)

    # Align lengths defensively
    n_points = points.shape[0]
    n_pose_vectors = pose_vectors.shape[0]
    n_mask = edge_mask.shape[0]
    n_common = min(n_points, n_pose_vectors, n_mask)

    if n_common == 0:
        print("[viz] No common rows to draw surface normals.")
        return

    # Get data for common indices
    P = points[:n_common]
    PV = pose_vectors[:n_common]
    EM = edge_mask[:n_common]

    # Count edge points
    n_edge_points = int(EM.sum())
    print(f"[viz] Found {n_edge_points} edge points for surface normal rendering")

    if n_edge_points == 0:
        print("[viz] No edge points found; skipping surface normal rendering.")
        return

    # Create arrows for edge points
    for i in range(n_common):
        if EM[i]:  # Only for edge points
            location = P[i]
            surface_normal = PV[i, 0, :]  # First row is surface normal

            # Normalize surface normal
            surface_normal = surface_normal / np.linalg.norm(surface_normal)

            # Create arrow
            arrow_normal = Arrow(
                location,
                location + surface_normal * arrow_length,
                c="gray",
            )
            arrow_normal.alpha(0.4)

            # Add to plotter and track for cleanup
            plotter.add(arrow_normal)
            surface_normal_arrows.append(arrow_normal)


def _add_mesh(plotter, mesh, mesh_translation, mesh_objects):
    """Add mesh visualization to the plotter.

    Args:
        plotter: Vedo plotter instance
        mesh: trimesh.Trimesh or trimesh.Scene object
        mesh_translation: [x,y,z] offset for the mesh (default [0,1.5,0])
        mesh_objects: list to store mesh objects for cleanup
    """
    if mesh is None:
        print("[viz] No mesh provided; skipping mesh rendering.")
        return

    # Set default translation if not provided
    if mesh_translation is None:
        mesh_translation = [0, 1.5, 0]

    # Handle single Trimesh object
    if hasattr(mesh, "vertices"):
        translated_vertices = mesh.vertices + mesh_translation
        vedo_mesh = Mesh([translated_vertices, mesh.faces])
        if hasattr(mesh.visual, "material") and hasattr(mesh.visual.material, "image"):
            vedo_mesh.alpha(0.4)
        else:
            vedo_mesh.color("lightblue").alpha(0.4)
        plotter.add(vedo_mesh)
        mesh_objects.append(vedo_mesh)
        print(f"[viz] Added mesh with {len(mesh.vertices)} vertices")

    # Handle Scene object with multiple parts
    elif hasattr(mesh, "geometry"):
        for geom_name, geom in mesh.geometry.items():
            translated_vertices = geom.vertices + mesh_translation
            vedo_mesh = Mesh([translated_vertices, geom.faces])
            if hasattr(geom.visual, "material") and hasattr(
                geom.visual.material, "image"
            ):
                vedo_mesh.alpha(0.4)
            else:
                vedo_mesh.color("lightblue").alpha(0.4)
            plotter.add(vedo_mesh)
            mesh_objects.append(vedo_mesh)
        print(f"[viz] Added mesh scene with {len(mesh.geometry)} geometries")
    else:
        print("[viz] Unknown mesh type; skipping mesh rendering.")


if __name__ == "__main__":
    # Set up paths
    pretrained_model_path = Path(
        "~/tbp/results/monty/pretrained_models/2d_sensor/"
        "disk_learning_standard_control/pretrained/model.pt"
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

            # Try to load the mesh
            mesh = load_mesh(object_name)
            if mesh is not None:
                # Handle both Scene and Trimesh objects
                if hasattr(mesh, "vertices"):
                    print(
                        f"  Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces"
                    )
                elif hasattr(mesh, "geometry"):
                    # Scene object - count total vertices/faces across all geometries
                    total_verts = sum(
                        len(geom.vertices) for geom in mesh.geometry.values()
                    )
                    total_faces = sum(
                        len(geom.faces) for geom in mesh.geometry.values()
                    )
                    print(
                        f"  Mesh loaded (Scene): {total_verts} vertices, {total_faces} faces"
                    )

            # Create interactive visualization
            visualize_point_cloud_interactive(
                model_data,
                title=f"Learned Point Cloud: {object_name}",
                mesh=mesh,
            )

        except Exception as e:
            print(f"  Error processing {object_name}: {e}")
            import traceback

            traceback.print_exc()
            continue
