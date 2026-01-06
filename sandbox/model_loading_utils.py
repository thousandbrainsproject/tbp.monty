"""Utility functions for loading model data from pretraining experiments."""

from pathlib import Path
import numpy as np
import torch


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
