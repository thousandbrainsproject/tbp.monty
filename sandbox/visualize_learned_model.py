"""Script to visualize the learned model using 2D Sensor Module."""

from pathlib import Path
import numpy as np
import torch
import trimesh
from vedo import Plotter, Points, Line, Text2D, Mesh


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
    graph_data = state_dict["lm_dict"][lm_id]["graph_memory"][object_name]["patch"]._graph

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
            feature_data = np.asarray(x_np[:, idx[0]: idx[1]])
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


def visualize_point_cloud_interactive(model_data, title=None, arrow_scale=0.002, mesh=None, mesh_translation=None):
    """Create interactive 3D visualization with Vedo.

    Args:
        model_data: Dictionary from load_object_model with points and features.
        title: Optional title for the plot.
        arrow_scale: Scale factor for edge orientation arrows.
        mesh: Optional trimesh.Trimesh object to display alongside point cloud.
        mesh_translation: Optional [x, y, z] translation to apply to mesh. Default is [0, 1.5, 0].
    """
    points = model_data["points"]
    features = model_data["features"]

    # Create plotter
    plotter = Plotter(size=(1400, 1000), title=title or "Learned Point Cloud")

    # Add mesh if provided
    if mesh is not None:
        if mesh_translation is None:
            mesh_translation = [0, 1.5, 0]

        # Handle both Scene and Trimesh objects
        if hasattr(mesh, 'vertices'):
            # Single Trimesh object
            translated_vertices = mesh.vertices + mesh_translation
            vedo_mesh = Mesh([translated_vertices, mesh.faces])

            # Apply texture if available
            if hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'image'):
                vedo_mesh.alpha(0.4)
            else:
                vedo_mesh.color("lightblue").alpha(0.4)

            plotter.add(vedo_mesh)
        elif hasattr(mesh, 'geometry'):
            # Scene object with multiple geometries
            for geom in mesh.geometry.values():
                translated_vertices = geom.vertices + mesh_translation
                vedo_mesh = Mesh([translated_vertices, geom.faces])

                # Apply texture if available
                if hasattr(geom.visual, 'material') and hasattr(geom.visual.material, 'image'):
                    vedo_mesh.alpha(0.4)
                else:
                    vedo_mesh.color("lightblue").alpha(0.4)

                plotter.add(vedo_mesh)

    # Determine coloring - prioritize edge_strength visualization
    if "edge_strength" in features:
        # Color points by edge strength
        edge_strengths = features["edge_strength"].flatten()
        # Normalize to [0, 1] for colormapping
        max_strength = np.percentile(edge_strengths, 95)  # Use 95th percentile to avoid outliers
        normalized_strength = np.clip(edge_strengths / max_strength, 0, 1)

        # Create point cloud with edge strength coloring
        point_cloud = Points(points, r=10)
        point_cloud.cmap("coolwarm", normalized_strength, vmin=0, vmax=1)
        point_cloud.add_scalarbar(title="Edge Strength")
        plotter.add(point_cloud)
    elif "rgba" in features and features["rgba"].shape[1] >= 3:
        # Use RGB colors from features
        # Vedo expects colors in range [0, 255] as a list of tuples or array
        colors = features["rgba"][:, :3].astype(np.uint8).tolist()
        point_cloud = Points(points, r=10)
        point_cloud.pointcolors = colors
        plotter.add(point_cloud)
    else:
        # Default gray color
        point_cloud = Points(points, r=10).color("gray")
        plotter.add(point_cloud)

    # Add edge tangent lines if available
    lines = []
    if "edge_tangent" in features:
        edge_tangents = features["edge_tangent"]

        # Get edge strength for filtering if available
        edge_strength_threshold = 0.1
        if "edge_strength" in features:
            edge_strengths = features["edge_strength"].flatten()
        else:
            edge_strengths = np.ones(len(points))  # Show all if no strength data

        # Create lines for points with detected edges
        for i in range(len(points)):
            # Check if edge is strong enough
            if edge_strengths[i] < edge_strength_threshold:
                continue

            tangent = edge_tangents[i]
            tangent_magnitude = np.linalg.norm(tangent)

            # Only show line if tangent is non-zero
            if tangent_magnitude > 1e-8:
                start_point = points[i]
                end_point = start_point + arrow_scale * tangent

                line = Line(
                    start_point,
                    end_point,
                    lw=0.1,
                    alpha=0.9
                )
                lines.append(line)
                plotter.add(line)

    # Add summary text with feature information
    info_lines = [
        f"Points: {len(points)}",
        f"Features: {', '.join(features.keys())}",
    ]
    if mesh is not None:
        # Handle both Scene and Trimesh objects
        if hasattr(mesh, 'vertices'):
            info_lines.append(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        elif hasattr(mesh, 'geometry'):
            total_verts = sum(len(geom.vertices) for geom in mesh.geometry.values())
            total_faces = sum(len(geom.faces) for geom in mesh.geometry.values())
            info_lines.append(f"Mesh (Scene): {total_verts} vertices, {total_faces} faces")
    if "edge_strength" in features:
        info_lines.append(
            f"Edge Strength: min={features['edge_strength'].min():.3f}, "
            f"max={features['edge_strength'].max():.3f}"
        )
    # if "edge_orientation" in features:
    #     info_lines.append(
    #         f"Edge Orientation: min={edge_angles.min():.3f}, "
    #         f"max={edge_angles.max():.3f} rad"
    #     )

    info_text = Text2D("\n".join(info_lines), pos="top-left", s=0.8, c="black")
    plotter.add(info_text)

    # Add legend for lines if edge tangents are shown
    if "edge_tangent" in features and len(lines) > 0:
        legend_text = Text2D(
            f"Yellow lines: Edge Tangents ({len(lines)} shown)",
            pos="bottom-left",
            s=0.8,
            c="yellow"
        )
        plotter.add(legend_text)

    # Set up axes and camera
    plotter.show(
        axes=dict(
            xtitle="X",
            ytitle="Y",
            ztitle="Z",
        ),
        viewup="y",
        camera=dict(
            pos=(0, 1.5, 0.2),      # Camera position: directly across from object at y=1.5, slightly back
            focal_point=(0, 1.5, 0),  # Look at center of object
            view_angle=45,  # Wider angle to see more
        ),
        interactive=True,
    )


if __name__ == "__main__":
    # Set up paths
    pretrained_model_path = Path(
        "~/tbp/results/monty/pretrained_models/pretrained_ycb_v10/"
        "supervised_pretraining_logos_2d_sensor/pretrained/model.pt"
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
            model_data = load_object_model(pretrained_model_path, object_name, lm_id=lm_id)

            print(f"  Points shape: {model_data['points'].shape}")
            print(f"  Available features: {list(model_data['features'].keys())}")

            # Try to load the mesh
            mesh = load_mesh(object_name)
            if mesh is not None:
                # Handle both Scene and Trimesh objects
                if hasattr(mesh, 'vertices'):
                    print(f"  Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                elif hasattr(mesh, 'geometry'):
                    # Scene object - count total vertices/faces across all geometries
                    total_verts = sum(len(geom.vertices) for geom in mesh.geometry.values())
                    total_faces = sum(len(geom.faces) for geom in mesh.geometry.values())
                    print(f"  Mesh loaded (Scene): {total_verts} vertices, {total_faces} faces")

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