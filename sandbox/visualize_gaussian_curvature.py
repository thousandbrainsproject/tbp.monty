"""Script to visualize learned models colored by Gaussian curvature."""

from pathlib import Path
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_object_model(model_path, object_name, lm_id=0):
    """Load an object model from a pretraining experiment."""
    state_dict = torch.load(model_path, map_location="cpu")
    graph_data = state_dict["lm_dict"][lm_id]["graph_memory"][object_name][
        f"patch_{lm_id}"
    ]._graph

    pos = getattr(graph_data, "pos", None)
    if pos is None and hasattr(graph_data, "__dict__"):
        pos = graph_data.__dict__.get("pos")
    if pos is None:
        raise RuntimeError("Expected attribute 'pos' on patch object")

    if isinstance(pos, torch.Tensor):
        points = pos.detach().cpu().numpy().astype(float)
    else:
        points = np.asarray(pos, dtype=float)

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
            feature_data = np.asarray(x_np[:, idx[0] : idx[1]])
            feature_dict[feature] = feature_data

    return {"points": points, "features": feature_dict}


def visualize_gaussian_curvature(model_data, object_name, output_path):
    """Create 3D scatter plot colored by Gaussian curvature and save to file.

    Args:
        model_data: dict with 'points' and 'features' keys.
        object_name: Name of the object for title.
        output_path: Path to save the PNG.

    Returns:
        Gaussian curvature values array.
    """
    points = np.asarray(model_data["points"], float)
    features = model_data["features"]

    if "gaussian_curvature" not in features:
        raise KeyError(f"gaussian_curvature not in features: {list(features.keys())}")

    gc = features["gaussian_curvature"].flatten()

    # Use 5th-95th percentile for color scale to avoid outlier distortion
    vmin = np.percentile(gc, 5)
    vmax = np.percentile(gc, 95)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=gc,
        cmap="coolwarm",
        s=15,
        alpha=0.8,
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Gaussian Curvature", fontsize=12)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"{object_name}\nGaussian Curvature", fontsize=14)

    # Equal aspect ratio
    max_range = (
        np.array([
            points[:, 0].max() - points[:, 0].min(),
            points[:, 1].max() - points[:, 1].min(),
            points[:, 2].max() - points[:, 2].min(),
        ]).max()
        / 2.0
    )
    mid = points.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return gc


def create_summary_plots(all_curvatures, object_names, output_dir):
    """Create histogram and box plot of all Gaussian curvatures."""
    # Flatten all curvatures for histogram
    all_gc = np.concatenate(all_curvatures)

    # Histogram
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(all_gc, bins=100, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Gaussian Curvature", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"Gaussian Curvature Distribution (All {len(object_names)} Objects)\n"
                 f"N={len(all_gc):,} points", fontsize=14)
    ax.axvline(0, color="red", linestyle="--", alpha=0.5, label="Zero")
    ax.legend()

    # Add statistics text
    stats_text = (
        f"Mean: {all_gc.mean():.4f}\n"
        f"Std: {all_gc.std():.4f}\n"
        f"Min: {all_gc.min():.4f}\n"
        f"Max: {all_gc.max():.4f}\n"
        f"Median: {np.median(all_gc):.4f}"
    )
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / "summary_histogram.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Box and whisker plot (single box across all objects)
    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot([all_gc], patch_artist=True)

    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)

    ax.set_xticklabels(["All Objects"])
    ax.set_ylabel("Gaussian Curvature", fontsize=12)
    ax.set_title(f"Gaussian Curvature Distribution ({len(object_names)} Objects)", fontsize=14)
    ax.axhline(0, color="red", linestyle="--", alpha=0.5, label="Zero")

    plt.tight_layout()
    plt.savefig(output_dir / "summary_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSummary statistics across all objects:")
    print(f"  Total points: {len(all_gc):,}")
    print(f"  Mean: {all_gc.mean():.6f}")
    print(f"  Std: {all_gc.std():.6f}")
    print(f"  Min: {all_gc.min():.6f}")
    print(f"  Max: {all_gc.max():.6f}")
    print(f"  Median: {np.median(all_gc):.6f}")


def process_model(model_path, output_dir):
    """Process a single model file and generate visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    print("Loading model...")
    state_dict = torch.load(model_path, map_location="cpu")
    lm_id = 0
    graph_memory = state_dict["lm_dict"][lm_id]["graph_memory"]
    available_objects = sorted(graph_memory.keys())
    print(f"Found {len(available_objects)} objects")

    print(f"\nProcessing {len(available_objects)} objects...")
    all_curvatures = []
    object_names = []

    for i, object_name in enumerate(available_objects):
        print(f"  [{i+1}/{len(available_objects)}] {object_name}")
        try:
            model_data = load_object_model(model_path, object_name, lm_id=lm_id)
            if "gaussian_curvature" not in model_data["features"]:
                print(f"    Skipping: no gaussian_curvature feature")
                continue

            gc = model_data["features"]["gaussian_curvature"].flatten()
            all_curvatures.append(gc)
            object_names.append(object_name)

            output_path = output_dir / f"{object_name}_gaussian_curvature.png"
            visualize_gaussian_curvature(model_data, object_name, output_path)
        except Exception as e:
            print(f"    Error: {e}")

    if all_curvatures:
        print("\nCreating summary plots...")
        create_summary_plots(all_curvatures, object_names, output_dir)

    print(f"\nDone! Results saved to: {output_dir}")
    return all_curvatures, object_names


if __name__ == "__main__":
    today = date.today().isoformat()

    model_configs = [
        # {
        #     "path": Path(
        #         "/Users/hlee/tbp/results/monty/pretrained_models/pretrained_ycb_v11/"
        #         "surf_agent_1lm_77obj/pretrained/model.pt"
        #     ),
        #     "name": "surf_agent_1lm_77obj",
        # },
        {
            "path": Path(
                "/Users/hlee/tbp/results/monty/pretrained_models/pretrained_ycb_v11/"
                "supervised_pre_training_curved_objects_after_flat_and_logo/pretrained/model.pt"
            ),
            "name": "logos_lvl4_comp_models",
        },
    ]

    for config in model_configs:
        model_path = config["path"]
        model_name = config["name"]
        print(f"\n{'='*60}")
        print(f"Processing: {model_name}")
        print(f"{'='*60}")

        output_dir = Path(f"results/{today}_gaussian_curvature/{model_name}")
        process_model(model_path, output_dir)
