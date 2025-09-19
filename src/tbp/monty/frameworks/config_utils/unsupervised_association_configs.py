# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Configuration utilities for unsupervised object ID association experiments.

This module provides helper functions and configuration classes for setting up
experiments that test unsupervised association learning between learning modules.
"""

import copy
from typing import Any, Dict, List, Optional

from tbp.monty.frameworks.models.evidence_matching.unsupervised_evidence_lm import (
    UnsupervisedEvidenceGraphLM,
)
from tbp.monty.frameworks.models.evidence_matching.unsupervised_model import (
    MontyForUnsupervisedAssociation,
)


def create_unsupervised_association_lm_config(
    base_lm_config: Dict,
    association_threshold: float = 0.1,
    min_association_threshold: float = 0.3,
    spatial_consistency_weight: float = 0.3,
    temporal_consistency_weight: float = 0.2,
    co_occurrence_weight: float = 0.5,
    max_association_memory_size: int = 1000,
    association_learning_enabled: bool = True,
) -> Dict:
    """Create a learning module configuration with unsupervised association.

    Args:
        base_lm_config: Base LM configuration dictionary to extend
        association_threshold: Minimum evidence threshold for recording associations
        min_association_threshold: Minimum association strength for vote mapping
        spatial_consistency_weight: Weight for spatial consistency in
            association strength
        temporal_consistency_weight: Weight for temporal consistency in
            association strength
        co_occurrence_weight: Weight for co-occurrence frequency in association
            strength
        max_association_memory_size: Maximum number of associations to keep in memory
        association_learning_enabled: Whether to enable association learning

    Returns:
        Dict: Enhanced LM configuration with association capabilities
    """
    # Create a copy of the base configuration
    enhanced_config = copy.deepcopy(base_lm_config)

    # Update the learning module class
    enhanced_config["learning_module_class"] = UnsupervisedEvidenceGraphLM

    # Add association-specific arguments
    if "learning_module_args" not in enhanced_config:
        enhanced_config["learning_module_args"] = {}

    enhanced_config["learning_module_args"].update(
        {
            "association_threshold": association_threshold,
            "min_association_threshold": min_association_threshold,
            "spatial_consistency_weight": spatial_consistency_weight,
            "temporal_consistency_weight": temporal_consistency_weight,
            "co_occurrence_weight": co_occurrence_weight,
            "max_association_memory_size": max_association_memory_size,
            "association_learning_enabled": association_learning_enabled,
        }
    )

    return enhanced_config


def create_cross_modal_lm_configs(
    base_lm_config: Dict,
    num_lms: int = 2,
    modality_names: Optional[List[str]] = None,
    association_params: Optional[Dict[str, Any]] = None,
) -> List[Dict]:
    """Create multiple LM configs for cross-modal association learning.

    Args:
        base_lm_config: Base LM configuration
        num_lms: Number of learning modules to create
        modality_names: Optional names for different modalities (e.g.,
            ['visual', 'touch'])
        association_params: Optional dictionary of association parameters

    Raises:
        ValueError: If the number of modality names does not match `num_lms`.

    Returns:
        List[Dict]: Learning module configurations for cross-modal learning
    """
    if association_params is None:
        association_params = {}

    if modality_names is None:
        modality_names = [f"modality_{i}" for i in range(num_lms)]
    elif len(modality_names) != num_lms:
        raise ValueError(
            f"Number of modality names ({len(modality_names)}) "
            f"must match number of LMs ({num_lms})"
        )

    lm_configs = []

    for i in range(num_lms):
        # Create enhanced configuration
        enhanced_config = create_unsupervised_association_lm_config(
            base_lm_config, **association_params
        )

        # Add modality-specific learning module ID
        if "learning_module_args" not in enhanced_config:
            enhanced_config["learning_module_args"] = {}

        enhanced_config["learning_module_args"]["learning_module_id"] = (
            f"{modality_names[i]}_lm_{i}"
        )

        lm_configs.append(enhanced_config)

    return lm_configs


def create_unsupervised_association_monty_config(
    base_monty_config: Dict,
    lm_configs: List[Dict],
    enable_association_analysis: bool = True,
    log_association_details: bool = False,
) -> Dict:
    """Create a Monty configuration with unsupervised association capabilities.

    Args:
        base_monty_config: Base Monty configuration dictionary
        lm_configs: List of learning module configuration dictionaries
            with association capabilities
        enable_association_analysis: Whether to enable cross-LM association analysis
        log_association_details: Whether to log detailed association information

    Returns:
        Enhanced Monty configuration dictionary with association capabilities
    """
    # Create a copy of the base configuration
    enhanced_config = copy.deepcopy(base_monty_config)

    # Update the Monty class
    enhanced_config["monty_class"] = MontyForUnsupervisedAssociation

    # Add association-specific arguments
    if "monty_args" not in enhanced_config:
        enhanced_config["monty_args"] = {}

    # Add custom arguments for the unsupervised association model
    enhanced_config["enable_association_analysis"] = enable_association_analysis
    enhanced_config["log_association_details"] = log_association_details

    # Update learning module configurations - convert list to dict format
    if isinstance(lm_configs, list):
        lm_config_dict = {}
        for i, lm_config in enumerate(lm_configs):
            lm_config_dict[f"learning_module_{i}"] = lm_config
        enhanced_config["learning_module_configs"] = lm_config_dict
    else:
        enhanced_config["learning_module_configs"] = lm_configs

    return enhanced_config


def create_simple_cross_modal_experiment_config(
    base_config: Dict[str, Any],
    modalities=None,
    association_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a simple cross-modal experiment configuration for association.

    Args:
        base_config: Base experiment configuration dictionary
        modalities: List of modality names
        association_params: Optional association learning parameters

    Returns:
        Dict[str, Any]: Enhanced experiment configuration with cross-modal association
            learning
    """
    if modalities is None:
        modalities = ["visual", "touch"]
    if association_params is None:
        association_params = {
            "association_threshold": 0.1,
            "min_association_threshold": 0.3,
            "spatial_consistency_weight": 0.3,
            "temporal_consistency_weight": 0.2,
            "co_occurrence_weight": 0.5,
        }

    # Create enhanced configuration
    enhanced_config = copy.deepcopy(base_config)

    # Extract base learning module configuration
    base_lm_config = enhanced_config["monty_config"]["learning_module_configs"][0]

    # Create cross-modal learning module configurations
    lm_configs = create_cross_modal_lm_configs(
        base_lm_config,
        num_lms=len(modalities),
        modality_names=modalities,
        association_params=association_params,
    )

    # Create enhanced Monty configuration
    enhanced_monty_config = create_unsupervised_association_monty_config(
        enhanced_config["monty_config"],
        lm_configs,
        enable_association_analysis=True,
        log_association_details=True,
    )

    enhanced_config["monty_config"] = enhanced_monty_config

    # Update experiment name to reflect association learning
    if "experiment_args" in enhanced_config:
        if "name" in enhanced_config["experiment_args"]:
            enhanced_config["experiment_args"]["name"] += "_unsupervised_association"

    return enhanced_config


# Predefined association parameter sets for different scenarios
ASSOCIATION_PARAMS_CONSERVATIVE = {
    "association_threshold": 0.2,
    "min_association_threshold": 0.5,
    "spatial_consistency_weight": 0.4,
    "temporal_consistency_weight": 0.3,
    "co_occurrence_weight": 0.3,
}

ASSOCIATION_PARAMS_AGGRESSIVE = {
    "association_threshold": 0.05,
    "min_association_threshold": 0.2,
    "spatial_consistency_weight": 0.2,
    "temporal_consistency_weight": 0.1,
    "co_occurrence_weight": 0.7,
}

ASSOCIATION_PARAMS_BALANCED = {
    "association_threshold": 0.1,
    "min_association_threshold": 0.3,
    "spatial_consistency_weight": 0.3,
    "temporal_consistency_weight": 0.2,
    "co_occurrence_weight": 0.5,
}


def get_association_params_preset(preset_name: str) -> Dict[str, Any]:
    """Get predefined association parameter sets.

    Args:
        preset_name: Name of the preset ('conservative', 'aggressive', 'balanced')

    Returns:
        Dict[str, Any]: Association parameters for the requested preset

    Raises:
        ValueError: If the preset name is not recognized.
    """
    presets = {
        "conservative": ASSOCIATION_PARAMS_CONSERVATIVE,
        "aggressive": ASSOCIATION_PARAMS_AGGRESSIVE,
        "balanced": ASSOCIATION_PARAMS_BALANCED,
    }

    if preset_name not in presets:
        available = list(presets.keys())
        raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")

    return copy.deepcopy(presets[preset_name])
