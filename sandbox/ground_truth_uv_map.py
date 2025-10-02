"""Script to visualize the ground truth UV map of the object.

The ground truth UV map is extracted from the object mesh via trimesh.
"""

import trimesh
import numpy as np
from matplotlib import pyplot as plt
import os
import glob
from pathlib import Path

def main():
    """Main function to visualize the ground truth UV map for all objects."""
    # Dataset configurations
    datasets = {
        "ycb": {
            "base_path": "/Users/hlee/tbp/data/habitat/versioned_data/ycb_1.2/meshes",
            "file_pattern": "google_16k/textured.glb.orig"
        },
        "compositional_objects": {
            "base_path": "/Users/hlee/tbp/data/compositional_objects/meshes",
            "file_pattern": "textured.glb"
        }
    }

    # Process both datasets
    for dataset_name, config in datasets.items():
        print(f"\n=== Processing {dataset_name} dataset ===")

        base_path = config["base_path"]
        file_pattern = config["file_pattern"]

        # Create dataset-specific results directory
        results_dir = Path(__file__).parent.parent / "results" / dataset_name
        results_dir.mkdir(parents=True, exist_ok=True)

        if not os.path.exists(base_path):
            print(f"Dataset path not found: {base_path}")
            continue

        # Find all object directories
        object_dirs = glob.glob(os.path.join(base_path, "*"))
        object_dirs = [d for d in object_dirs if os.path.isdir(d)]

        print(f"Found {len(object_dirs)} objects to process")

        for obj_dir in sorted(object_dirs):
            object_name = os.path.basename(obj_dir)
            file_path = os.path.join(obj_dir, file_pattern)

            if not os.path.exists(file_path):
                print(f"Skipping {object_name}: file not found")
                continue

            print(f"Processing {object_name}...")

            try:
                # Load the mesh
                with open(file_path, "rb") as f:
                    scene = trimesh.load_mesh(f, file_type="glb")

                # Create visualization
                create_visualization(scene, object_name, results_dir, dataset_name)

            except Exception as e:
                print(f"Error processing {object_name}: {e}")
                continue

def create_visualization(scene, object_name, results_dir, dataset_name):
    """Create and save visualization for a single object."""
    uv_map = scene.visual.uv
    color, texture_image = extract_color(scene)

    # Create four side-by-side plots
    fig = plt.figure(figsize=(24, 5))

    vertices = scene.vertices
    faces = scene.faces
    color_normalized = color[:, :3] / 255.0  # Use RGB only, ignore alpha

    # Plot 1: 3D Mesh with colors
    ax1 = fig.add_subplot(141, projection='3d')
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # Create triangular patches
    triangles_3d = vertices[faces]
    face_colors = np.mean(color_normalized[faces], axis=1)

    mesh_collection = Poly3DCollection(triangles_3d,
                                      facecolors=face_colors,
                                      alpha=0.8,
                                      edgecolors='none')
    ax1.add_collection3d(mesh_collection)

    # Set consistent axis limits for both 3D plots
    x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()

    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_zlim(z_min, z_max)
    ax1.set_title(f"3D Mesh - {dataset_name.upper()}: {object_name}")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # Plot 2: 3D Point Cloud
    ax2 = fig.add_subplot(142, projection='3d')
    ax2.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                c=color_normalized, s=3)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_zlim(z_min, z_max)
    ax2.set_title(f"3D Point Cloud - {dataset_name.upper()}: {object_name}")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    # Plot 3: UV Map
    ax3 = fig.add_subplot(143)
    ax3.scatter(uv_map[:, 0], uv_map[:, 1], c=color_normalized, s=3)
    ax3.set_title(f"UV Map - {dataset_name.upper()}: {object_name}")
    ax3.set_xlabel("U")
    ax3.set_ylabel("V")
    ax3.set_aspect('equal')

    # Plot 4: Texture Image
    ax4 = fig.add_subplot(144)
    ax4.imshow(texture_image)
    ax4.set_title(f"Texture Image - {dataset_name.upper()}: {object_name}")

    ax4.axis('off')

    plt.tight_layout()

    # Save the plot
    output_path = results_dir / f"{object_name}_uv_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close to free memory
    print(f"Saved: {output_path}")

def extract_color(mesh: trimesh.Trimesh) -> np.ndarray:
    """Extract the color from the object mesh."""

    # For textured meshes, sample colors from the texture using UV coordinates
    if (hasattr(mesh.visual, 'material') and
        mesh.visual.material is not None and
        hasattr(mesh.visual.material, 'baseColorTexture')):

        print("Sampling colors from texture")
        texture_image = np.array(mesh.visual.material.baseColorTexture)
        uv_coords = mesh.visual.uv

        print(f"Texture shape: {texture_image.shape}")
        print(f"UV coords shape: {uv_coords.shape}")

        # Convert UV coordinates to pixel coordinates
        # UV coordinates are typically in [0,1] range
        height, width = texture_image.shape[:2]
        u_pixels = (uv_coords[:, 0] * (width - 1)).astype(int)
        v_pixels = ((1 - uv_coords[:, 1]) * (height - 1)).astype(int)  # Flip V coordinate

        # Clamp to valid range
        u_pixels = np.clip(u_pixels, 0, width - 1)
        v_pixels = np.clip(v_pixels, 0, height - 1)

        # Sample colors from texture
        if len(texture_image.shape) == 3:  # RGB or RGBA
            color = texture_image[v_pixels, u_pixels]
            # Add alpha channel if not present
            if color.shape[1] == 3:
                alpha = np.full((color.shape[0], 1), 255, dtype=color.dtype)
                color = np.concatenate([color, alpha], axis=1)
        else:  # Grayscale
            gray = texture_image[v_pixels, u_pixels]
            color = np.stack([gray, gray, gray, np.full_like(gray, 255)], axis=1)

    # Fallback to other color methods
    elif hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        print("Using vertex colors")
        color = mesh.visual.vertex_colors
    elif hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
        print("Using face colors")
        color = mesh.visual.face_colors
    else:
        print("Using default gray")
        color = np.full((len(mesh.vertices), 4), [128, 128, 128, 255], dtype=np.uint8)

    print(f"Final color shape: {color.shape}")
    return color, texture_image

if __name__ == "__main__":
    main()