"""Script to visualize the learned model points only (no edges)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from vedo import Points, Plotter

from model_loading_utils import load_object_model


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
        points[:, 1],
        points[:, 0],
        points[:, 2],
        c=colors,
        s=20,
        alpha=0.6,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title or "Learned Point Cloud (Matplotlib)")

    # Set equal aspect ratio
    max_range = (
        np.array(
            [
                points[:, 0].max() - points[:, 0].min(),
                points[:, 1].max() - points[:, 1].min(),
                points[:, 2].max() - points[:, 2].min(),
            ]
        ).max()
        / 2.0
    )
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

    # Calculate camera position based on point cloud bounds
    center = points.mean(axis=0)
    # Position camera at a distance proportional to the size of the point cloud
    max_range = np.array(
        [
            points[:, 0].max() - points[:, 0].min(),
            points[:, 1].max() - points[:, 1].min(),
            points[:, 2].max() - points[:, 2].min(),
        ]
    ).max()
    camera_distance = max_range * 1.5

    # ----- Axes & camera -----
    camera_pos = (
        center[0],
        center[1] + camera_distance,
        center[2] + camera_distance * 0.3,
    )
    plotter.show(
        axes=dict(xtitle="X", ytitle="Y", ztitle="Z"),
        viewup="y",
        camera=dict(
            pos=camera_pos,
            focal_point=center,
            view_angle=0,
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
