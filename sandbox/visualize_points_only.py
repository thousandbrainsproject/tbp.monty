"""Script to visualize the learned model points only (no edges)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from vedo import Arrows, Points, Plotter

from model_loading_utils import load_object_model


def extract_principal_curvature_directions(features, points):
    """Extract and validate principal curvature directions from pose_vectors.

    Args:
        features: dict of features from model_data
        points: (N, 3) array of point positions

    Returns:
        tuple: (dir1, dir2, valid_mask) where:
            - dir1: (N, 3) first principal direction vectors
            - dir2: (N, 3) second principal direction vectors
            - valid_mask: (N,) boolean array indicating valid directions
    """
    if "pose_vectors" not in features:
        return None, None, None

    # Extract pose_vectors
    pv = np.asarray(features["pose_vectors"], float)

    # Reshape from (N, 9) to (N, 3, 3) if needed
    if pv.ndim == 2 and pv.shape[1] == 9:
        pv = pv.reshape(-1, 3, 3)

    # Align sizes
    n_points = points.shape[0]
    n_pose = pv.shape[0]
    n_common = min(n_points, n_pose)

    if n_common == 0:
        return None, None, None

    # Extract dir1 and dir2 (indices 1 and 2 in the 3x3 matrix)
    # pose_vectors structure: [surface_normal, dir1, dir2]
    dir1 = pv[:n_common, 1, :]  # First principal direction
    dir2 = pv[:n_common, 2, :]  # Second principal direction

    # Validate: check for finite, non-zero vectors
    dir1_valid = np.isfinite(dir1).all(axis=1) & (np.linalg.norm(dir1, axis=1) > 1e-9)
    dir2_valid = np.isfinite(dir2).all(axis=1) & (np.linalg.norm(dir2, axis=1) > 1e-9)
    valid_mask = dir1_valid & dir2_valid

    # Normalize directions
    dir1_norm = np.linalg.norm(dir1, axis=1, keepdims=True)
    dir2_norm = np.linalg.norm(dir2, axis=1, keepdims=True)
    dir1_normalized = np.where(dir1_norm > 1e-9, dir1 / dir1_norm, dir1)
    dir2_normalized = np.where(dir2_norm > 1e-9, dir2 / dir2_norm, dir2)

    print(f"[basis_arrows] Found pose_vectors for {n_common} points")
    print(f"[basis_arrows] Valid directions: {valid_mask.sum()} / {n_common}")

    return dir1_normalized, dir2_normalized, valid_mask


def visualize_point_cloud_matplotlib(
    model_data,
    title=None,
):
    """Create matplotlib 3D scatter plot visualization.

    Args:
        model_data: dict with keys:
            - points: (N,3) world coords
            - features: dict. Expected options:
                * optional 'rgba': (N,3 or 4)
        title: window title.
    """
    points = np.asarray(model_data["points"], float)
    features = model_data["features"]

    # Print point cloud statistics
    print(f"[matplotlib] Points shape: {points.shape}")
    print("[matplotlib] Point bounds:")
    print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    print(f"  Center: {points.mean(axis=0)}")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Get colors
    if "rgba" in features:
        colors = features["rgba"][:, :3] / 255.0  # Normalize to [0, 1]
    else:
        colors = "gray"

    # Plot points
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors,
        s=20,
        alpha=0.6,
    )

    # Calculate arrow length as 2% of bounding box diagonal
    max_range = np.array(
        [
            points[:, 0].max() - points[:, 0].min(),
            points[:, 1].max() - points[:, 1].min(),
            points[:, 2].max() - points[:, 2].min(),
        ]
    ).max()
    arrow_length = max_range * 0.02

    # Extract and visualize principal curvature directions
    dir1, dir2, valid_mask = extract_principal_curvature_directions(features, points)
    if dir1 is not None and valid_mask is not None and valid_mask.sum() > 0:
        valid_idx = np.where(valid_mask)[0]
        n_valid = len(valid_idx)
        
        # Randomly sample a subset of points for arrows (max 50 points)
        max_arrows = min(50, n_valid)
        if n_valid > max_arrows:
            sampled_idx = np.random.choice(n_valid, size=max_arrows, replace=False)
            sampled_valid_idx = valid_idx[sampled_idx]
        else:
            sampled_valid_idx = valid_idx
        
        sampled_points = points[sampled_valid_idx]
        sampled_dir1 = dir1[sampled_valid_idx]
        sampled_dir2 = dir2[sampled_valid_idx]

        # Draw dir1 arrows in red
        ax.quiver(
            sampled_points[:, 0],
            sampled_points[:, 1],
            sampled_points[:, 2],
            sampled_dir1[:, 0] * arrow_length,
            sampled_dir1[:, 1] * arrow_length,
            sampled_dir1[:, 2] * arrow_length,
            color="red",
            arrow_length_ratio=0.3,
            alpha=0.7,
            linewidth=1.5,
            label="dir1 (first principal direction)",
        )

        # Draw dir2 arrows in blue
        ax.quiver(
            sampled_points[:, 0],
            sampled_points[:, 1],
            sampled_points[:, 2],
            sampled_dir2[:, 0] * arrow_length,
            sampled_dir2[:, 1] * arrow_length,
            sampled_dir2[:, 2] * arrow_length,
            color="blue",
            arrow_length_ratio=0.3,
            alpha=0.7,
            linewidth=1.5,
            label="dir2 (second principal direction)",
        )
        ax.legend()
        print(f"[matplotlib] Added basis arrows for {len(sampled_valid_idx)} randomly sampled points (out of {n_valid} valid)")
    else:
        print("[matplotlib] No valid pose_vectors found; skipping basis arrows")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title or "Learned Point Cloud (Matplotlib)")

    # Set equal aspect ratio
    max_range = max_range / 2.0
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()


def visualize_point_cloud_interactive(
    model_data,
    title=None,
):
    """Create interactive 3D visualization with Vedo (points only, no edges).

    Args:
        model_data: dict with keys:
            - points: (N,3) world coords
            - features: dict. Expected options:
                * optional 'rgba': (N,3 or 4)
        title: window title.
    """
    points = np.asarray(model_data["points"], float)
    features = model_data["features"]

    # Debug: print available features
    print(f"[vedo] Available features: {list(features.keys())}")
    print(f"[vedo] Points shape: {points.shape}")
    print("[vedo] Point bounds:")
    print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    print(f"  Center: {points.mean(axis=0)}")

    plotter = Plotter(size=(1400, 1000), title=title or "Learned Point Cloud")

    # ----- Point colors -----
    point_cloud_obj = Points(points, r=10)

    # Use rgba if available, otherwise default to gray
    if "rgba" in features:
        colors = features["rgba"][:, :3].astype(np.uint8).tolist()
        point_cloud_obj.pointcolors = colors
    else:
        point_cloud_obj.color("gray")

    plotter.add(point_cloud_obj)

    # Calculate arrow length as 2% of bounding box diagonal
    max_range = np.array(
        [
            points[:, 0].max() - points[:, 0].min(),
            points[:, 1].max() - points[:, 1].min(),
            points[:, 2].max() - points[:, 2].min(),
        ]
    ).max()
    arrow_length = max_range * 0.02

    # Extract and visualize principal curvature directions
    dir1_arrows = None
    dir2_arrows = None
    arrows_visible = True
    
    dir1, dir2, valid_mask = extract_principal_curvature_directions(features, points)
    if dir1 is not None and valid_mask is not None and valid_mask.sum() > 0:
        valid_idx = np.where(valid_mask)[0]
        n_valid = len(valid_idx)
        
        # Randomly sample a subset of points for arrows (max 50 points)
        max_arrows = min(50, n_valid)
        if n_valid > max_arrows:
            sampled_idx = np.random.choice(n_valid, size=max_arrows, replace=False)
            sampled_valid_idx = valid_idx[sampled_idx]
        else:
            sampled_valid_idx = valid_idx
        
        sampled_points = points[sampled_valid_idx]
        sampled_dir1 = dir1[sampled_valid_idx]
        sampled_dir2 = dir2[sampled_valid_idx]

        # Create arrow endpoints for dir1 (red)
        dir1_starts = sampled_points
        dir1_ends = sampled_points + sampled_dir1 * arrow_length
        dir1_arrows = Arrows(
            dir1_starts, dir1_ends, c="red", alpha=0.7
        )
        dir1_arrows.lw(2)  # Set line width after creation
        plotter.add(dir1_arrows)

        # Create arrow endpoints for dir2 (blue)
        dir2_starts = sampled_points
        dir2_ends = sampled_points + sampled_dir2 * arrow_length
        dir2_arrows = Arrows(
            dir2_starts, dir2_ends, c="blue", alpha=0.7
        )
        dir2_arrows.lw(2)  # Set line width after creation
        plotter.add(dir2_arrows)

        print(f"[vedo] Added basis arrows for {len(sampled_valid_idx)} randomly sampled points (out of {n_valid} valid)")
        
        # Add toggle button for arrows
        def toggle_arrows(button, _event):
            nonlocal arrows_visible
            if dir1_arrows is not None and dir2_arrows is not None:
                if arrows_visible:
                    dir1_arrows.off()
                    dir2_arrows.off()
                    arrows_visible = False
                else:
                    dir1_arrows.on()
                    dir2_arrows.on()
                    arrows_visible = True
                plotter.render()
        
        plotter.add_button(
            toggle_arrows,
            pos=(0.85, 0.05),
            states=["Hide Arrows", "Show Arrows"],
            size=20,
            font="Calco",
        )
        print("[vedo] Added toggle button for arrows (arrows visible by default)")
    else:
        print("[vedo] No valid pose_vectors found; skipping basis arrows")

    # Calculate camera position based on point cloud bounds
    center = points.mean(axis=0)
    # Position camera at a distance proportional to the size of the point cloud
    camera_distance = max_range * 1.5

    # ----- Axes & camera -----
    camera_pos = (
        center[0],
        center[1] + camera_distance,
        center[2] + camera_distance * 0.3,
    )
    plotter.show(
        axes=dict(xtitle="X", ytitle="Y", ztitle="Z"),
        viewup="x",
        camera=dict(
            pos=camera_pos,
            focal_point=center,
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

            # Show matplotlib visualization first
            print("\nShowing matplotlib visualization...")
            visualize_point_cloud_matplotlib(
                model_data,
                title=f"Learned Point Cloud: {object_name}",
            )

            # Then show interactive vedo visualization
            print("\nShowing vedo interactive visualization...")
            visualize_point_cloud_interactive(
                model_data,
                title=f"Learned Point Cloud: {object_name}",
            )

        except Exception as e:
            print(f"  Error processing {object_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

