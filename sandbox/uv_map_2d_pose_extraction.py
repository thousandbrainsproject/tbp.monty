"""Script to extract 2D poses from UV map texture patches using structure tensor.

This script loads object meshes and extracts 2D pose information (edge directions)
from texture patches in the UV map using the structure tensor approach.
The structure tensor provides robust edge direction estimation by analyzing
local gradient patterns in a 9x9 neighborhood around each point.
"""

import trimesh
import numpy as np
from matplotlib import pyplot as plt
import os
import glob
from pathlib import Path
from skimage.color import rgb2gray
from skimage.filters import gaussian, sobel_h, sobel_v
from scipy import ndimage
from scipy.linalg import eigh
import cv2


def main():
    """Main function to extract 2D poses for all objects."""
    # Dataset configurations
    datasets = {
        "ycb": {
            "base_path": "/Users/hlee/tbp/data/habitat/versioned_data/ycb_1.2/meshes",
            "file_pattern": "google_16k/textured.glb.orig",
        },
        "compositional_objects": {
            "base_path": "/Users/hlee/tbp/data/compositional_objects/meshes",
            "file_pattern": "textured.glb",
        },
    }

    # Process both datasets
    for dataset_name, config in datasets.items():
        print(f"\n=== Processing {dataset_name} dataset ===")

        base_path = config["base_path"]
        file_pattern = config["file_pattern"]

        # Create dataset-specific results directory
        results_dir = (
            Path(__file__).parent.parent / "results" / dataset_name / "2d_poses"
        )
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

                # Extract 2D poses and create visualization
                extract_and_visualize_2d_poses(
                    scene, object_name, results_dir, dataset_name
                )

            except Exception as e:
                print(f"Error processing {object_name}: {e}")
                continue


def extract_texture_patch(texture_image, uv_coord, patch_size=64):
    """Extract a texture patch centered at UV coordinate.

    Args:
        texture_image: RGB/RGBA texture image of shape (H, W, C)
        uv_coord: UV coordinate [u, v] in [0, 1] range
        patch_size: Size of the patch to extract (default: 64x64)

    Returns:
        RGB patch of shape (patch_size, patch_size, 3) or None if invalid
    """
    if texture_image is None or len(texture_image.shape) < 3:
        return None

    height, width = texture_image.shape[:2]

    # Convert UV to pixel coordinates
    u_pixel = int(uv_coord[0] * (width - 1))
    v_pixel = int((1 - uv_coord[1]) * (height - 1))  # Flip V coordinate

    # Calculate patch boundaries
    half_patch = patch_size // 2
    u_start = max(0, u_pixel - half_patch)
    u_end = min(width, u_pixel + half_patch)
    v_start = max(0, v_pixel - half_patch)
    v_end = min(height, v_pixel + half_patch)

    # Extract patch
    patch = texture_image[v_start:v_end, u_start:u_end]

    # Ensure we have RGB only (drop alpha if present)
    if patch.shape[2] == 4:
        patch = patch[:, :, :3]

    # Resize to exact patch size if needed
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        from skimage.transform import resize

        patch = resize(
            patch, (patch_size, patch_size), preserve_range=True, anti_aliasing=True
        )
        patch = patch.astype(np.uint8)

    return patch


def center_coords(h, w):
    """Get center coordinates of patch."""
    return h // 2, w // 2  # (row, col); for 64×64 → (32,32)


def gradient_to_tangent_angle(gradient_angle):
    """Convert gradient direction to edge tangent direction and wrap to [0, 2π).

    Args:
        gradient_angle: Gradient direction in radians (any range)

    Returns:
        Edge tangent angle in [0, 2π) radians
    """
    # Edge tangent is perpendicular to gradient (rotate by 90°)
    tangent_angle = gradient_angle + np.pi / 2
    # Wrap to [0, 2π)
    return (tangent_angle + 2 * np.pi) % (2 * np.pi)


def sobel_center_strength_and_angle(patch):
    """Simple Sobel edge detection at center pixel."""
    # Convert to BGR for OpenCV (if RGB)
    if len(patch.shape) == 3:
        img_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)

    # Gray luminance normalized to [0, 1]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    r, c = center_coords(*gray.shape)
    gx, gy = Ix[r, c], Iy[r, c]
    mag = np.hypot(gx, gy)
    gradient_ang = np.arctan2(gy, gx)  # gradient direction, [-pi, pi]
    tangent_ang = gradient_to_tangent_angle(gradient_ang)  # edge tangent, [0, 2π)
    return float(mag), float(tangent_ang)


def dicenzo_color_edge_at_center(patch):
    """DiZenzo color edge detection at center pixel."""
    # Convert to BGR for OpenCV (if RGB)
    if len(patch.shape) == 3:
        img_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)

    # Compute per-channel Sobel, normalized to [0, 1]
    f = img_bgr.astype(np.float32) / 255.0
    Ix = np.stack(
        [cv2.Sobel(f[..., i], cv2.CV_32F, 1, 0, ksize=3) for i in range(3)], axis=-1
    )
    Iy = np.stack(
        [cv2.Sobel(f[..., i], cv2.CV_32F, 0, 1, ksize=3) for i in range(3)], axis=-1
    )

    # DiZenzo structure matrix entries
    Gxx = np.sum(Ix * Ix, axis=-1)
    Gyy = np.sum(Iy * Iy, axis=-1)
    Gxy = np.sum(Ix * Iy, axis=-1)

    r, c = center_coords(*img_bgr.shape[:2])
    gxx, gyy, gxy = float(Gxx[r, c]), float(Gyy[r, c]), float(Gxy[r, c])

    # Largest eigenvalue (edge strength^2) and orientation
    disc = np.sqrt((gxx - gyy) ** 2 + 4.0 * (gxy**2))
    lam_max = 0.5 * (gxx + gyy + disc)  # strength^2
    strength = np.sqrt(max(lam_max, 0.0))
    gradient_theta = 0.5 * np.arctan2(
        2.0 * gxy, (gxx - gyy + 1e-12)
    )  # gradient direction
    tangent_theta = gradient_to_tangent_angle(gradient_theta)  # edge tangent, [0, 2π)
    return strength, float(tangent_theta)


def structure_tensor_center(patch, win_sigma=1.0, ksize=7):
    """Enhanced structure tensor at center with OpenCV implementation."""
    # Convert to BGR for OpenCV (if RGB)
    if len(patch.shape) == 3:
        img_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)

    # Gradients on luminance, normalized to [0, 1], then smoothed outer-products in a window
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    Jxx = Ix * Ix
    Jyy = Iy * Iy
    Jxy = Ix * Iy

    Jxx = cv2.GaussianBlur(Jxx, (ksize, ksize), win_sigma)
    Jyy = cv2.GaussianBlur(Jyy, (ksize, ksize), win_sigma)
    Jxy = cv2.GaussianBlur(Jxy, (ksize, ksize), win_sigma)

    r, c = center_coords(*gray.shape)
    jxx, jyy, jxy = float(Jxx[r, c]), float(Jyy[r, c]), float(Jxy[r, c])

    disc = np.sqrt((jxx - jyy) ** 2 + 4.0 * (jxy**2))
    lam1 = 0.5 * (jxx + jyy + disc)
    lam2 = 0.5 * (jxx + jyy - disc)
    edge_strength = np.sqrt(max(lam1, 0.0))
    coherence = (lam1 - lam2) / (
        lam1 + lam2 + 1e-12
    )  # ~[0,1], 1=edge-like, 0=flat/noisy
    gradient_theta = 0.5 * np.arctan2(
        2.0 * jxy, (jxx - jyy + 1e-12)
    )  # gradient direction
    tangent_theta = gradient_to_tangent_angle(gradient_theta)  # edge tangent, [0, 2π)
    return edge_strength, coherence, float(tangent_theta)


def center_canny_hit_with_strength(patch, low=30, high=100, radius=3, sigma=1.0):
    """Canny edge detection with comparable gradient magnitude strength."""
    # Convert to gray, normalized to [0, 1]
    if len(patch.shape) == 3:
        img_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Pre-smooth explicitly (OpenCV's Canny does not expose sigma cleanly)
    if sigma and sigma > 0:
        gray_blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    else:
        gray_blur = gray

    # Gradient magnitude like Canny uses (L2) - on normalized [0,1] image
    Ix = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(Ix, Iy)  # sqrt(Ix^2 + Iy^2)

    # Run Canny on the same blurred image for consistency - convert back to uint8 for Canny
    edges = cv2.Canny((gray_blur * 255).astype(np.uint8), low, high, L2gradient=True)

    # Local window around center
    h, w = gray.shape
    r, c = h // 2, w // 2
    r0, r1 = max(0, r - radius), min(h, r + radius + 1)
    c0, c1 = max(0, c - radius), min(w, c + radius + 1)

    # Binary hit if any Canny edge is within the window
    hit = (edges[r0:r1, c0:c1] > 0).any()

    # Comparable "Canny strength": max gradient magnitude among Canny-accepted pixels in the window
    window_edges = edges[r0:r1, c0:c1] > 0
    if window_edges.any():
        local_mag = float(grad_mag[r0:r1, c0:c1][window_edges].max())
    else:
        local_mag = 0.0

    # Optional normalized score (0..1) by the high threshold
    norm_score = min(local_mag / float(high), 1.0) if high > 0 else 0.0

    # Compute gradient angle at center for edge direction
    gradient_ang = np.arctan2(Iy[r, c], Ix[r, c])  # gradient direction
    tangent_ang = gradient_to_tangent_angle(gradient_ang)  # edge tangent, [0, 2π)

    return {
        "has_edge": bool(hit),
        "edges": edges,
        "strength": local_mag,  # comparable magnitude (same units as Sobel magnitude)
        "norm_strength01": norm_score,
        "angle": float(tangent_ang),  # edge tangent angle in [0, 2π)
    }


def center_canny_hit(patch, low=50, high=150, radius=2):
    """Legacy Canny function for backward compatibility."""
    result = center_canny_hit_with_strength(patch, low, high, radius)
    return result["has_edge"], result["edges"]


def compute_structure_tensor(patch, window_size=9, sigma=1.0):
    """Original structure tensor implementation (kept for comparison)."""
    # Convert to grayscale
    if len(patch.shape) == 3:
        gray_patch = rgb2gray(patch)
    else:
        gray_patch = patch

    # Apply Gaussian smoothing
    if sigma > 0:
        gray_patch = gaussian(gray_patch, sigma=sigma, preserve_range=True)

    # Compute gradients
    grad_x = sobel_v(gray_patch)  # Vertical edges (horizontal gradient)
    grad_y = sobel_h(gray_patch)  # Horizontal edges (vertical gradient)

    # Compute structure tensor components
    Ixx = grad_x * grad_x
    Iyy = grad_y * grad_y
    Ixy = grad_x * grad_y

    # Apply Gaussian weighting in the window
    if window_size > 1:
        window_sigma = window_size / 6.0  # Standard choice
        Ixx = gaussian(Ixx, sigma=window_sigma, preserve_range=True)
        Iyy = gaussian(Iyy, sigma=window_sigma, preserve_range=True)
        Ixy = gaussian(Ixy, sigma=window_sigma, preserve_range=True)

    # Get center values for the structure tensor
    center = patch.shape[0] // 2
    tensor = np.array(
        [
            [Ixx[center, center], Ixy[center, center]],
            [Ixy[center, center], Iyy[center, center]],
        ]
    )

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigh(tensor)

    # Sort by eigenvalue magnitude (largest first)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Compute coherence measure
    if eigenvalues[0] + eigenvalues[1] > 1e-10:
        coherence = (eigenvalues[0] - eigenvalues[1]) / (
            eigenvalues[0] + eigenvalues[1]
        )
    else:
        coherence = 0.0

    return {
        "tensor": tensor,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "coherence": coherence,
        "gradients": (grad_x[center, center], grad_y[center, center]),
    }


def get_edge_direction(structure_tensor_result):
    """Extract edge direction from structure tensor eigenanalysis.

    Args:
        structure_tensor_result: Result from compute_structure_tensor

    Returns:
        Dictionary containing:
            - edge_direction: Edge tangent direction in radians [0, 2π]
            - edge_strength: Magnitude of the dominant eigenvalue
            - edge_confidence: Coherence measure indicating edge quality
            - has_edge: Boolean indicating if a significant edge exists
    """
    eigenvalues = structure_tensor_result["eigenvalues"]
    eigenvectors = structure_tensor_result["eigenvectors"]
    coherence = structure_tensor_result["coherence"]

    # The eigenvector with the largest eigenvalue points in the direction
    # of maximum intensity change (gradient direction)
    # The edge direction is perpendicular to this (rotate 90 degrees)
    gradient_direction = eigenvectors[
        :, 0
    ]  # First column = largest eigenvalue eigenvector

    # Edge tangent is perpendicular to gradient direction
    edge_tangent = np.array([-gradient_direction[1], gradient_direction[0]])

    # Convert to angle in radians [0, 2π]
    edge_direction = np.arctan2(edge_tangent[1], edge_tangent[0])
    if edge_direction < 0:
        edge_direction += 2 * np.pi

    # Edge strength is the largest eigenvalue
    edge_strength = eigenvalues[0]

    # Use coherence as confidence measure
    edge_confidence = coherence

    # Determine if edge exists based on eigenvalue ratio and strength
    has_edge = (
        edge_strength > 0.01  # Minimum strength threshold
        and coherence > 0.1  # Minimum coherence threshold
    )

    return {
        "edge_direction": edge_direction,
        "edge_strength": edge_strength,
        "edge_confidence": edge_confidence,
        "has_edge": has_edge,
        "gradient_direction": gradient_direction,
        "edge_tangent": edge_tangent,
    }


def detect_edge_presence(patch, variance_threshold=0.001):
    """Determine if an edge exists in the patch (not plain color).

    Args:
        patch: RGB patch of shape (H, W, 3)
        variance_threshold: Minimum variance to consider non-plain (in [0,1] range)

    Returns:
        Dictionary containing:
            - is_plain: Boolean indicating if patch is mostly uniform color
            - variance: Color variance in the patch
            - mean_color: Average color in the patch
    """
    # Convert to grayscale for variance calculation
    if len(patch.shape) == 3:
        gray_patch = rgb2gray(patch)
    else:
        gray_patch = patch

    # Calculate variance
    variance = np.var(gray_patch)

    # Calculate mean color
    if len(patch.shape) == 3:
        mean_color = np.mean(patch, axis=(0, 1))
    else:
        mean_color = np.mean(gray_patch)

    # Determine if plain
    is_plain = variance < variance_threshold

    return {"is_plain": is_plain, "variance": variance, "mean_color": mean_color}


def sample_texture_grid(texture_image, patch_size=64, stride=32):
    """Sample patches directly from texture image in a grid pattern.

    Args:
        texture_image: RGB/RGBA texture image of shape (H, W, C)
        patch_size: Size of patches to extract
        stride: Stride between patch centers

    Returns:
        List of (patch, texture_coord) tuples where texture_coord is (row, col)
    """
    if texture_image is None:
        return []

    height, width = texture_image.shape[:2]
    patches = []

    # Grid sampling across texture
    for row in range(patch_size // 2, height - patch_size // 2, stride):
        for col in range(patch_size // 2, width - patch_size // 2, stride):
            # Extract patch centered at (row, col)
            r_start = row - patch_size // 2
            r_end = row + patch_size // 2
            c_start = col - patch_size // 2
            c_end = col + patch_size // 2

            patch = texture_image[r_start:r_end, c_start:c_end]

            # Ensure we have RGB only (drop alpha if present)
            if patch.shape[2] == 4:
                patch = patch[:, :, :3]

            # Ensure exact patch size
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                # Convert texture coordinates to UV coordinates for compatibility
                uv_coord = np.array([col / width, 1.0 - row / height])  # Flip V
                patches.append((patch, (row, col), uv_coord))

    print(f"Sampled {len(patches)} patches from texture grid")
    return patches


def compare_edge_detection_methods(patch):
    """Compare multiple edge detection methods on the same patch.

    Args:
        patch: RGB patch of shape (H, W, 3)

    Returns:
        Dictionary with results from all methods
    """
    results = {}

    # Method 1: Simple Sobel
    try:
        strength, angle = sobel_center_strength_and_angle(patch)
        results["sobel"] = {
            "strength": strength,
            "angle": angle,
            "has_edge": strength > 2.0,  # Will use normalized threshold later
            "method": "sobel",
        }
    except Exception as e:
        results["sobel"] = {"error": str(e)}

    # Method 2: DiZenzo color edge
    try:
        strength, angle = dicenzo_color_edge_at_center(patch)
        results["dicenzo"] = {
            "strength": strength,
            "angle": angle,
            "has_edge": strength > 5.0,  # Will use normalized threshold later
            "method": "dicenzo",
        }
    except Exception as e:
        results["dicenzo"] = {"error": str(e)}

    # Method 3: Enhanced structure tensor
    try:
        strength, coherence, angle = structure_tensor_center(
            patch, win_sigma=1.0, ksize=7
        )
        results["structure_tensor_enhanced"] = {
            "strength": strength,
            "coherence": coherence,
            "angle": angle,
            "has_edge": strength > 2.0
            and coherence > 0.05,  # Will use normalized threshold later
            "method": "structure_tensor_enhanced",
        }
    except Exception as e:
        results["structure_tensor_enhanced"] = {"error": str(e)}

    # Method 4: Canny edge detection with proper gradient magnitude
    try:
        canny_result = center_canny_hit_with_strength(patch, low=30, high=100, radius=3)
        results["canny"] = {
            "has_edge": canny_result["has_edge"],
            "edges": canny_result["edges"],
            "method": "canny",
            "strength": canny_result[
                "strength"
            ],  # Actual gradient magnitude, not pixel count
            "angle": canny_result["angle"],  # Edge tangent angle
        }
    except Exception as e:
        results["canny"] = {"error": str(e)}

    # Method 5: Original structure tensor (for comparison)
    try:
        tensor_result = compute_structure_tensor(patch, window_size=9, sigma=1.0)
        eigenvalues = tensor_result["eigenvalues"]
        coherence = tensor_result["coherence"]
        edge_strength = eigenvalues[0] if len(eigenvalues) > 0 else 0.0

        results["structure_tensor_original"] = {
            "strength": np.sqrt(max(edge_strength, 0.0)),  # Fix: sqrt of eigenvalue
            "coherence": coherence,
            "has_edge": np.sqrt(max(edge_strength, 0.0)) > 0.03
            and coherence > 0.05,  # Adjusted threshold
            "method": "structure_tensor_original",
        }
    except Exception as e:
        results["structure_tensor_original"] = {"error": str(e)}

    return results


def normalize_edge_strengths_per_method(pose_results):
    """Normalize edge strengths per-method across all patches.

    Each method gets normalized independently to [0, 1] range based on its own
    distribution of strengths across all patches.

    Args:
        pose_results: List of all pose results (modified in-place)
    """
    # Collect strengths by method across all patches
    method_strengths = {}

    for result in pose_results:
        for method_name, method_result in result["edge_methods"].items():
            if (
                isinstance(method_result, dict)
                and "strength" in method_result
                and "error" not in method_result
            ):
                if method_name not in method_strengths:
                    method_strengths[method_name] = []
                method_strengths[method_name].append(method_result["strength"])

    # Compute per-method percentiles for normalization
    method_scales = {}
    for method_name, strengths in method_strengths.items():
        if strengths:
            max_strength = np.percentile(strengths, 95)
            min_strength = np.percentile(strengths, 5)
            method_scales[method_name] = {
                "min": min_strength,
                "max": max_strength,
                "range": max_strength - min_strength,
            }

    # Normalize each method's values using its own scale
    for result in pose_results:
        for method_name, method_result in result["edge_methods"].items():
            if isinstance(method_result, dict) and "error" not in method_result:
                if method_name in method_scales and "strength" in method_result:
                    scale = method_scales[method_name]
                    raw = method_result["strength"]

                    if scale["range"] > 1e-10:
                        normalized = (raw - scale["min"]) / scale["range"]
                    else:
                        normalized = 0.5  # Default if all strengths are equal

                    method_result["normalized_strength"] = np.clip(normalized, 0, 1)

                    # Apply normalized thresholds consistently (lowered to detect more edges)
                    if method_name == "structure_tensor_enhanced":
                        coherence = method_result.get("coherence", 0.0)
                        method_result["has_edge"] = (
                            normalized > 0.3 and coherence > 0.05
                        )
                    elif method_name in ["sobel", "dicenzo", "canny"]:
                        method_result["has_edge"] = normalized > 0.3
                    elif method_name == "structure_tensor_original":
                        coherence = method_result.get("coherence", 0.0)
                        method_result["has_edge"] = (
                            normalized > 0.2 and coherence > 0.05
                        )
                    else:
                        method_result["has_edge"] = normalized > 0.3
                else:
                    # Missing strength or scale
                    method_result["normalized_strength"] = 0.0
                    method_result["has_edge"] = False
            else:
                # Error case
                if isinstance(method_result, dict):
                    method_result["normalized_strength"] = 0.0
                    method_result["has_edge"] = False


def select_best_edge_examples(pose_results, n_examples=3):
    """Select the best edge examples for visualization.

    Args:
        pose_results: List of pose extraction results
        n_examples: Number of examples to select

    Returns:
        List of best examples sorted by edge quality
    """
    # Filter valid poses (any method detecting an edge)
    valid_poses = []
    for r in pose_results:
        has_any_edge = any(
            method_result.get("has_edge", False)
            for method_result in r.get("edge_methods", {}).values()
            if not isinstance(method_result, dict) or "error" not in method_result
        )
        if has_any_edge:
            valid_poses.append(r)

    if not valid_poses:
        return []

    # Sort by maximum normalized edge strength across all methods for fair comparison
    def edge_quality(result):
        max_normalized_strength = 0.0
        edge_methods = result.get("edge_methods", {})
        for method_result in edge_methods.values():
            if (
                isinstance(method_result, dict)
                and "normalized_strength" in method_result
            ):
                max_normalized_strength = max(
                    max_normalized_strength, method_result["normalized_strength"]
                )

        variance = result["edge_presence"]["variance"]
        return max_normalized_strength * np.log(1 + variance)

    sorted_poses = sorted(valid_poses, key=edge_quality, reverse=True)
    return sorted_poses[:n_examples]


def sample_edges_spatially(valid_poses, texture_shape, grid_size=20, max_edges=5000):
    """Sample edges spatially across texture to avoid clustering.

    Args:
        valid_poses: List of edge detection results
        texture_shape: (height, width) of texture
        grid_size: Number of grid cells per dimension
        max_edges: Maximum number of edges to return

    Returns:
        Subset of valid_poses spatially distributed
    """
    if not valid_poses:
        return []

    height, width = texture_shape[:2]
    cell_height = height // grid_size
    cell_width = width // grid_size

    # Group edges by grid cell
    grid_cells = {}
    for pose in valid_poses:
        texture_coord = pose.get("texture_coord", (0, 0))
        row, col = texture_coord

        # Determine grid cell
        cell_row = min(row // cell_height, grid_size - 1)
        cell_col = min(col // cell_width, grid_size - 1)
        cell_key = (cell_row, cell_col)

        if cell_key not in grid_cells:
            grid_cells[cell_key] = []
        grid_cells[cell_key].append(pose)

    # Select strongest edge from each cell
    sampled_edges = []
    for cell_edges in grid_cells.values():
        if cell_edges:
            # Sort by normalized edge strength and take the best
            best_edge = max(
                cell_edges, key=lambda x: x["edge_result"]["normalized_strength"]
            )
            sampled_edges.append(best_edge)

    # Sort by normalized strength and limit to max_edges
    sampled_edges.sort(
        key=lambda x: x["edge_result"]["normalized_strength"], reverse=True
    )
    return sampled_edges[:max_edges]


def draw_2d_pose_on_patch(
    patch,
    edge_direction,
    tangent_color=(255, 255, 0),
    normal_color=(0, 255, 255),
    arrow_length=20,
):
    """Draw both tangent and normal arrows to show 2D pose.

    Args:
        patch: RGB patch of shape (H, W, 3)
        edge_direction: Edge tangent direction in radians
        tangent_color: RGB color for tangent arrow (default: yellow)
        normal_color: RGB color for normal arrow (default: cyan)
        arrow_length: Length of arrows in pixels

    Returns:
        Patch with 2D pose arrows drawn on it
    """
    # Create a copy to avoid modifying original
    patch_with_pose = patch.copy()

    # Center of patch
    center_y, center_x = patch.shape[0] // 2, patch.shape[1] // 2

    # Tangent arrow (edge direction)
    tangent_end_x = int(center_x + arrow_length * np.cos(edge_direction))
    tangent_end_y = int(center_y + arrow_length * np.sin(edge_direction))

    # Normal arrow (perpendicular to edge, 90 degree rotation)
    normal_direction = edge_direction + np.pi / 2
    normal_length = arrow_length * 0.7  # Slightly shorter
    normal_end_x = int(center_x + normal_length * np.cos(normal_direction))
    normal_end_y = int(center_y + normal_length * np.sin(normal_direction))

    # Draw tangent arrow (edge direction)
    cv2.arrowedLine(
        patch_with_pose,
        (center_x, center_y),
        (tangent_end_x, tangent_end_y),
        tangent_color,
        thickness=3,
        tipLength=0.3,
    )

    # Draw normal arrow (perpendicular)
    cv2.arrowedLine(
        patch_with_pose,
        (center_x, center_y),
        (normal_end_x, normal_end_y),
        normal_color,
        thickness=3,
        tipLength=0.3,
    )

    # Draw center point
    cv2.circle(patch_with_pose, (center_x, center_y), 4, (255, 255, 255), -1)

    # Remove T/N labels to avoid clutter - use color differentiation only

    return patch_with_pose


def draw_arrow_on_patch(
    patch, edge_direction, arrow_color=(255, 255, 0), arrow_length=20
):
    """Draw a single arrow on a texture patch to show edge direction.

    This is kept for backward compatibility. Use draw_2d_pose_on_patch for full 2D pose.
    """
    # Create a copy to avoid modifying original
    patch_with_arrow = patch.copy()

    # Center of patch
    center_y, center_x = patch.shape[0] // 2, patch.shape[1] // 2

    # Calculate arrow end point
    end_x = int(center_x + arrow_length * np.cos(edge_direction))
    end_y = int(center_y + arrow_length * np.sin(edge_direction))

    # Draw arrow line
    cv2.arrowedLine(
        patch_with_arrow,
        (center_x, center_y),
        (end_x, end_y),
        arrow_color,
        thickness=3,
        tipLength=0.3,
    )

    # Draw center point
    cv2.circle(patch_with_arrow, (center_x, center_y), 3, arrow_color, -1)

    return patch_with_arrow


def extract_and_visualize_2d_poses(scene, object_name, results_dir, dataset_name):
    """Extract 2D poses and create visualization for a single object."""
    # Extract texture and UV map
    uv_map = scene.visual.uv
    texture_image = extract_texture_from_scene(scene)

    if texture_image is None:
        print(f"No texture found for {object_name}")
        return

    print(f"Texture shape: {texture_image.shape}, UV map shape: {uv_map.shape}")

    # NEW APPROACH: Sample directly from texture image grid
    patch_samples = sample_texture_grid(texture_image, patch_size=64, stride=32)

    if not patch_samples:
        print("No patches could be extracted from texture")
        return

    print(f"Processing {len(patch_samples)} texture patches of size 64x64 pixels")

    # Extract 2D poses for all texture patches
    pose_results = []
    valid_poses = []

    for i, (patch, texture_coord, uv_coord) in enumerate(patch_samples):
        # Check if patch has meaningful content (variance in [0,1] range)
        edge_presence = detect_edge_presence(patch, variance_threshold=0.001)

        # Apply ALL edge detection methods and compare
        edge_methods = compare_edge_detection_methods(patch)

        # Note: Normalization will be done per-method across all patches later
        # Store raw results for now

        # Combine results (without normalized values yet)
        result = {
            "uv_coord": uv_coord,
            "texture_coord": texture_coord,
            "patch": patch,
            "edge_presence": edge_presence,
            "edge_methods": edge_methods,  # All methods for comparison (raw values)
            "processing_index": i,
        }

        pose_results.append(result)

    # Normalize edge strengths per-method across all patches
    print("Normalizing edge strengths per method...")
    normalize_edge_strengths_per_method(pose_results)

    # Now determine best method and valid poses after normalization
    valid_poses = []
    for result in pose_results:
        # Determine the best method using normalized strengths
        best_method = None
        best_normalized_strength = 0.0
        best_raw_strength = 0.0
        best_angle = 0.0
        has_any_edge = False

        for method_name, method_result in result["edge_methods"].items():
            if "error" in method_result:
                continue

            if method_result.get("has_edge", False):
                has_any_edge = True
                normalized_strength = method_result.get("normalized_strength", 0.0)
                if normalized_strength > best_normalized_strength:
                    best_normalized_strength = normalized_strength
                    best_raw_strength = method_result.get("strength", 0.0)
                    best_method = method_name
                    best_angle = method_result.get("angle", 0.0)

        # Create edge result for visualization
        result["edge_result"] = {
            "has_edge": has_any_edge,
            "edge_strength": best_raw_strength,
            "normalized_strength": best_normalized_strength,
            "edge_direction": best_angle,
            "edge_confidence": best_normalized_strength if has_any_edge else 0.0,
            "best_method": best_method,
        }

        if has_any_edge:
            valid_poses.append(result)

    print(f"Extracted {len(pose_results)} total poses, {len(valid_poses)} valid poses")

    # Print method statistics
    method_counts = {}
    for result in valid_poses:
        best_method = result["edge_result"]["best_method"]
        if best_method:
            method_counts[best_method] = method_counts.get(best_method, 0) + 1

    print("Edge detection method performance:")
    for method, count in sorted(method_counts.items()):
        print(f"  {method}: {count} detections")

    # Select best examples for visualization
    best_examples = select_best_edge_examples(pose_results, n_examples=3)
    print(f"Selected {len(best_examples)} best edge examples")

    # Create visualization
    create_pose_visualization(
        scene,
        object_name,
        results_dir,
        dataset_name,
        pose_results,
        valid_poses,
        texture_image,
        best_examples,
    )


def extract_texture_from_scene(scene):
    """Extract texture image from trimesh scene."""
    try:
        if (
            hasattr(scene.visual, "material")
            and scene.visual.material is not None
            and hasattr(scene.visual.material, "baseColorTexture")
        ):
            return np.array(scene.visual.material.baseColorTexture)
        else:
            return None
    except Exception as e:
        print(f"Error extracting texture: {e}")
        return None


def create_pose_visualization(
    scene,
    object_name,
    results_dir,
    dataset_name,
    pose_results,
    valid_poses,
    texture_image,
    best_examples,
):
    """Create focused visualization of 2D poses with UV map and texture analysis."""
    # Create figure with 2x3 layout
    fig = plt.figure(figsize=(18, 12))

    uv_map = scene.visual.uv

    # Plot 1: Clean texture image
    ax1 = fig.add_subplot(231)
    ax1.imshow(texture_image)
    ax1.set_title(
        f"Texture Image\n{texture_image.shape[0]}x{texture_image.shape[1]} pixels"
    )
    ax1.axis("off")

    # Plot 2: Texture with patch locations
    ax2 = fig.add_subplot(232)
    ax2.imshow(texture_image, alpha=0.7)

    # Overlay sampled patch centers to avoid overcrowding
    display_patches = sample_edges_spatially(
        pose_results, texture_image.shape, grid_size=25, max_edges=1000
    )
    # display_patches = sample_edges_spatially(valid_poses, texture_image.shape, grid_size=25, max_edges=200)
    for result in display_patches:
        texture_coord = result.get("texture_coord", (0, 0))
        strength = result["edge_result"]["normalized_strength"]
        has_edge = result["edge_result"]["has_edge"]

        color = "red" if has_edge else "blue"
        marker_size = max(
            15, min(60, strength * 60)
        )  # Scale marker size (strength is now [0,1])
        alpha = 0.9 if has_edge else 0.4

        ax2.scatter(
            texture_coord[1],
            texture_coord[0],
            c=color,
            s=marker_size,
            alpha=alpha,
            edgecolors="white",
            linewidth=1,
        )

    ax2.set_title(
        f"Texture with Patch Locations\n{len(pose_results)} patches, {len(valid_poses)} with edges"
    )
    ax2.axis("off")

    # Plot 3: Edge strength distribution by method (moved from plot 4)
    ax3 = fig.add_subplot(233)
    if valid_poses:
        # Collect normalized strengths by method
        method_strengths = {}
        for result in valid_poses:
            method = result["edge_result"]["best_method"]
            strength = result["edge_result"]["normalized_strength"]
            if method not in method_strengths:
                method_strengths[method] = []
            method_strengths[method].append(strength)

        # Create histogram
        colors = ["red", "green", "blue", "orange", "purple"]
        method_colors = {
            "sobel": "red",
            "dicenzo": "green",
            "structure_tensor_enhanced": "blue",
            "canny": "orange",
            "structure_tensor_original": "purple",
        }

        bins = np.linspace(0, 1, 30)  # Normalized strength is always [0, 1]

        for i, (method, strengths) in enumerate(method_strengths.items()):
            color = method_colors.get(method, colors[i % len(colors)])
            ax3.hist(
                strengths,
                bins=bins,
                alpha=0.6,
                label=method.replace("_", " "),
                color=color,
                density=True,
            )

        ax3.set_xlabel("Normalized Edge Strength")
        ax3.set_ylabel("Normalized Density")
        ax3.set_title("Normalized Edge Strength Distribution by Method")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(
            0.5,
            0.5,
            "No edge data\navailable",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax3.transAxes,
        )
        ax3.axis("off")

    # Bottom row: Examples 1, 2, and 3 (plots 4, 5, 6)
    if best_examples and len(best_examples) >= 3:
        # Show all 3 examples
        for i, example in enumerate(best_examples[:3]):
            ax_idx = 234 + i  # 234, 235, 236
            ax = fig.add_subplot(ax_idx)

            # Get patch and edge information
            patch = example["patch"]
            edge_dir = example["edge_result"]["edge_direction"]
            strength = example["edge_result"]["normalized_strength"]
            best_method = example["edge_result"]["best_method"]

            # Draw 2D pose on patch (tangent + normal arrows)
            patch_with_pose = draw_2d_pose_on_patch(patch, edge_dir)

            ax.imshow(patch_with_pose)
            ax.set_title(
                f"Example {i + 1}: {best_method}\nNorm. Strength={strength:.2f}",
                fontsize=10,
            )
            ax.axis("off")

    elif best_examples and len(best_examples) >= 2:
        # Show 2 examples + method performance
        for i, example in enumerate(best_examples[:2]):
            ax_idx = 234 + i  # 234, 235
            ax = fig.add_subplot(ax_idx)

            # Get patch and edge information
            patch = example["patch"]
            edge_dir = example["edge_result"]["edge_direction"]
            strength = example["edge_result"]["normalized_strength"]
            best_method = example["edge_result"]["best_method"]

            # Draw 2D pose on patch (tangent + normal arrows)
            patch_with_pose = draw_2d_pose_on_patch(patch, edge_dir)

            ax.imshow(patch_with_pose)
            ax.set_title(
                f"Example {i + 1}: {best_method}\nNorm. Strength={strength:.2f}",
                fontsize=10,
            )
            ax.axis("off")

        # Plot method performance in third slot
        ax6 = fig.add_subplot(236)
        method_counts = {}
        for result in valid_poses:
            best_method = result["edge_result"]["best_method"]
            if best_method:
                method_counts[best_method] = method_counts.get(best_method, 0) + 1

        if method_counts:
            methods = list(method_counts.keys())
            counts = list(method_counts.values())
            ax6.bar(range(len(methods)), counts)
            ax6.set_xticks(range(len(methods)))
            ax6.set_xticklabels(
                [m.replace("_", "\n") for m in methods], rotation=45, ha="right"
            )
            ax6.set_title(f"Method Performance\n{sum(counts)} total detections")
            ax6.set_ylabel("Detections")
        else:
            ax6.text(
                0.5,
                0.5,
                "No methods\ndetected edges",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax6.transAxes,
            )
            ax6.axis("off")

    else:
        # Show debugging information if no good examples
        for i in range(3):
            ax = fig.add_subplot(234 + i)
            ax.text(
                0.5,
                0.5,
                f"No edge\nexample {i + 1}",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
            )
            ax.axis("off")

    plt.tight_layout()

    # Save the plot
    output_path = results_dir / f"{object_name}_2d_poses.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
