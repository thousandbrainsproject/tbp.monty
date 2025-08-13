- Start Date: 2025-07-04
- RFC PR:

# RFC: Enhanced Non-Morphological Feature Extraction in Sensor Modules

## Summary

This RFC proposes enhancing non-morphological feature extraction in Monty's sensor modules by implementing multi-scale
Local Binary Patterns (LBP) for texture analysis and enhanced depth processing for tactile-like features. The
implementation uses a **pure composition approach** with a dedicated texture processor that can be integrated into any
sensor module, improving object recognition and discrimination capabilities while maintaining rotation invariance as
specified in the project documentation. This approach avoids multiple inheritance complexity and aligns with established
architectural preferences in the codebase.

## Motivation

Monty's current non-morphological feature extraction is limited to single-pixel sampling, extracting only RGBA/HSV
values from the center pixel of a 64x64 patch (using just 1/4096 available pixels). This approach cannot distinguish
between smooth and textured surfaces with identical colors, lacks rotation invariance, and provides only basic depth
statistics without surface texture information.

The [Extract Better Features](https://thousandbrainsproject.readme.io/docs/extract-better-features) documentation
specifically calls for richer features using Local Binary Patterns for visual features and enhanced depth processing to
approximate tactile texture, with an explicit requirement for **rotation invariant** features that detect textured
patterns regardless of sensor orientation.

The main research question is how much improvement we can achieve with enhanced texture features, and how robust these
features are to noise, scale variations, and different viewing conditions. Secondary questions include optimal
integration of RGB and depth texture features, acceptable computational overhead, and correlation with existing
morphological features.


## Detailed Design

### Architecture Overview

To enable sensor modules to extract enhanced texture features, we propose a compositional approach where each sensor
module owns feature-specific data processors that can be configured and utilized as needed. These processors serve as
simple interfaces to standalone functions that don't require sensor module attributes or methods, allowing us to avoid
complex inheritance patterns.

The implementation consists of standalone texture processing utilities, dedicated feature processors for LBP and
depth-based features (kept separate for modularity), and clean integration into sensor modules through composition
rather than inheritance. This approach works with any RGBA+depth data and maintains compatibility with existing sensor
module patterns.

### 1. LBP Texture Processing

The LBP processing module provides configurable multi-scale rotation-invariant texture analysis using Local Binary Patterns. The system is designed to be easily reconfigurable for different radius and point combinations - we propose starting with radii [1, 2] pixels and point counts [8, 16] as reasonable defaults based on literature, but these parameters should be systematically tested and optimized for different object types and scenarios.

Key capabilities include fully configurable multi-scale analysis, robust input validation, normalized histogram generation, and graceful handling of edge cases. The module generates descriptive feature names like `lbp_texture_r1_p8` and `lbp_texture_r2_p16` that reflect the specific parameter combinations used, making it easy to experiment with different configurations.

*Implementation details: See [Appendix A: LBP Processing Implementation](#appendix-a-lbp-processing-implementation)*

### 2. Depth Texture Processing

The depth texture processing module extracts tactile-like features from surface depth variations to approximate texture information that would be available through touch. It computes surface roughness (scaled standard deviation of depth), gradient magnitude (rate of depth change), and surface normal consistency (alignment with expected surface orientation).

The implementation includes robust handling of edge cases such as empty patches, insufficient on-object points, and non-square patch geometries. It uses Gaussian smoothing before gradient computation to reduce noise and employs careful NaN handling to prevent runtime warnings. Features are generated with names like `depth_texture_roughness`, `depth_texture_gradient`, and `depth_texture_consistency`.

*Implementation details: See [Appendix B: Depth Texture Processing Implementation](#appendix-b-depth-texture-processing-implementation)*

#### Key Design Principles

Both LBP and depth texture processing follow a modular design with standalone functions that can be easily tested and
integrated into any sensor module with RGBA+depth data. LBP provides multi-scale analysis with rotation invariance
through uniform patterns, while depth processing extracts tactile-like surface characteristics including roughness,
gradient patterns, and surface normal consistency.

### 3. Feature Processors

We implement separate processors for LBP and depth-based features to enable independent usage and avoid coupling
unrelated functionality.

#### LBP Feature Processor

The LBP processor provides configurable multiscale texture analysis with preset configurations (basic, enhanced,
complete) to simplify adoption. It handles feature name generation, extraction coordination, and maintains compatibility
with existing sensor module patterns.

#### Depth Texture Feature Processor

The depth processor extracts tactile-like features from surface variations, including roughness metrics, gradient
analysis, and surface normal consistency checking. It provides robust handling of edge cases and invalid data points.

### 4. Integration Approach

Sensor modules integrate texture processing through composition, instantiating the appropriate processors (LBP and/or
depth) based on their feature requirements. The processors provide a clean interface for feature extraction,
configuration management, and compatibility with existing noise systems.

The integration maintains backward compatibility by making texture features completely opt-in, with existing
configurations continuing to work unchanged. Feature extraction is performed only when texture features are explicitly
requested, avoiding computational overhead for users who don't need enhanced texture analysis.

## Implementation Plan

The implementation follows a phased approach focusing on core infrastructure first, then integration, validation, and
optimization.

### Phase 1: Core Infrastructure

Create standalone texture processing utilities and feature processors with comprehensive testing. Implement LBP
processing utilities for multi-scale texture analysis and depth texture processing for tactile-like features.

### Phase 2: Sensor Module Integration

Integrate texture processors into sensor modules using composition, maintaining backward compatibility and following
existing patterns from NoiseMixin integration.

### Phase 3: Configuration and Validation

Develop configuration presets and comprehensive validation experiments, including benchmark integration with existing
YCB experiments for performance comparison.

### Phase 4: Advanced Features and Optimization

Performance optimization, advanced texture features, and comprehensive documentation with usage examples.

## Evaluation Criteria

The success of enhanced texture features will be measured through texture discrimination capability, rotation
invariance, integration compatibility, computational efficiency, and parameter optimization effectiveness. Key metrics include >90%
accuracy on texture discrimination tasks, <5% feature variation across rotations, and ≥5% improvement on YCB recognition
benchmarks.

Validation will include controlled texture experiments, systematic parameter optimization studies (testing different radius combinations like [1,3], [1,2,3], [2,4]), benchmark integration tests with existing YCB datasets,
performance analysis, and feature correlation studies to ensure texture features complement rather than interfere with
existing morphological features.

## Alternatives Considered

**HTM Spatial Pooler**: While aligned with TBT principles and providing adaptive learning, it has high implementation
complexity and no inherent rotation invariance. Decision: implement as complementary approach in Phase 4.

**Gabor Filters**: Provide multi-scale analysis but lack rotation invariance and have high dimensionality. LBP provides
better rotation invariance with lower computational cost.

**Deep Learning Feature Extractors**: Offer state-of-the-art performance but conflict with TBP philosophy and have high
computational cost. Explicitly avoided per project FAQ.

**Architecture Alternatives**: Direct integration vs. composition approach. Composition selected for better
maintainability, code reuse, and cleaner architecture despite slightly more complex initialization.

## Risks and Mitigations

**Performance Regression**: New features might hurt performance where texture isn't discriminative. Mitigated through
hybrid approach maintaining existing features, configurable weights, and gradual rollout.

**Computational Overhead**: Multi-scale processing may increase processing time. Mitigated through efficient LBP
implementation, configurable parameters, and parallel processing opportunities.

**Feature Dimensionality**: Multi-scale features may create high-dimensional vectors. Mitigated through configurable
feature selection and importance analysis.

**Configuration Complexity**: Many parameters may complicate usage. Mitigated through preset configurations, simplified
feature names, and smart defaults.

## Questions for Community

**Technical Design**: Is the separate LBP and depth processor approach optimal? How should we handle processor lifecycle
and configuration complexity?

**Integration**: Should texture features be completely opt-in or gradually enabled by default? What are acceptable
computational overhead limits?

**Validation**: Which experiments should be prioritized - YCB recognition, texture discrimination tasks, or real-world
iPad experiments?

**Research Direction**: Should HTM Spatial Pooler be implemented in parallel with LBP or deferred until LBP is
validated?

## References

**Primary Documentation
**: [Extract Better Features](https://thousandbrainsproject.readme.io/docs/extract-better-features), [TBP FAQ on Deep Learning](https://thousandbrainsproject.readme.io/docs/faq-monty#why-does-monty-not-make-use-of-deep-learning)

**Technical References**: Ojala et al. (2002) "Multiresolution gray-scale and rotation invariant texture classification
with local binary patterns", scikit-image LBP implementation

**Related Work**: PR #343 (reference standard for experimental analysis), existing sensor modules and noise handling patterns

---

## Appendix A: LBP Processing Implementation

#### New File: `src/tbp/monty/frameworks/utils/lbp_processing.py`

```python
"""Local Binary Pattern texture processing utilities for sensor modules."""

import numpy as np
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray


def extract_multiscale_lbp_features(rgba_patch, config=None):
    """Extract multi-scale rotation-invariant Local Binary Pattern texture features.

    Args:
        rgba_patch: RGBA patch data (H x W x 4)
        config: Configuration dict with 'radii', 'n_points', 'method', 'channels'

    Returns:
        dict: Multi-scale LBP histogram features
    """
    if config is None:
        config = {
            "radii": [1, 2],
            "n_points": [8, 16],
            "method": "uniform",
            "channels": "grayscale"
        }

    # Validate input patch
    if rgba_patch is None or rgba_patch.size == 0:
        return {}

    if len(rgba_patch.shape) != 3 or rgba_patch.shape[2] < 3:
        raise ValueError("rgba_patch must be H x W x 4 (or at least H x W x 3)")

    features = {}

    # Convert to grayscale for texture analysis
    if config["channels"] == "grayscale":
        gray_patch = rgb2gray(rgba_patch[:, :, :3])
        channels = [gray_patch]
        channel_names = [""]
    elif config["channels"] == "rgb":
        channels = [rgba_patch[:, :, i] for i in range(3)]
        channel_names = ["_r", "_g", "_b"]
    else:  # individual channels
        channels = [rgba_patch[:, :, i] for i in range(3)]
        channel_names = ["_r", "_g", "_b"]

    for ch_idx, (channel, ch_name) in enumerate(zip(channels, channel_names)):
        for r_idx, (radius, n_points) in enumerate(zip(config["radii"], config["n_points"])):
            # Extract rotation-invariant uniform LBP
            lbp = local_binary_pattern(channel, P=n_points, R=radius, method=config["method"])

            # Create normalized histogram
            n_bins = n_points + 2 if config["method"] == "uniform" else 2**n_points
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
            # Normalize histogram to create probability distribution
            hist_sum = hist.sum()
            normalized_hist = hist / hist_sum if hist_sum > 0 else np.zeros_like(hist)

            feature_name = f"lbp_texture{ch_name}_r{radius}_p{n_points}"
            features[feature_name] = normalized_hist

    return features
```

## Appendix B: Depth Texture Processing Implementation

#### New File: `src/tbp/monty/frameworks/utils/depth_texture_processing.py`

```python
"""Depth-based texture processing utilities for sensor modules."""

import numpy as np
from scipy.ndimage import gaussian_filter


def extract_depth_texture_features(depth_patch, obs_3d, surface_normal=None, config=None):
    """Extract enhanced tactile-like features from depth variations.

    Args:
        depth_patch: Depth values for the patch (flattened)
        obs_3d: 3D observation data with object mask (N x 4)
        surface_normal: Optional surface normal vector for consistency checking
        config: Configuration dict with processing parameters

    Returns:
        dict: Enhanced depth-based texture features
    """
    if config is None:
        config = {
            "roughness_scale": 1.0,
            "gradient_kernel_size": 3,
            "normalize": True,
            "surface_consistency_threshold": 0.1
        }

    # Filter to on-object points only
    on_object_mask = obs_3d[:, 3] > 0
    if not np.any(on_object_mask):
        return {
            "depth_texture_roughness": 0.0,
            "depth_texture_gradient": 0.0,
            "depth_texture_consistency": 0.0
        }

    on_object_depths = depth_patch[on_object_mask]

    # Handle edge case of insufficient data points
    if len(on_object_depths) < 2:
        return {
            "depth_texture_roughness": 0.0,
            "depth_texture_gradient": 0.0,
            "depth_texture_consistency": 0.0
        }

    # Surface roughness (scaled standard deviation of depth)
    # Use ddof=0 to avoid "Mean of empty slice" warnings with small sample sizes
    depth_roughness = np.std(on_object_depths, ddof=0) * config["roughness_scale"]

    # Enhanced depth gradient analysis
    patch_size = int(np.sqrt(len(depth_patch)))
    if patch_size * patch_size != len(depth_patch):
        # Handle non-square patches by using the closest square size
        patch_size = int(np.ceil(np.sqrt(len(depth_patch))))
        padded_depth = np.zeros(patch_size * patch_size)
        padded_depth[:len(depth_patch)] = depth_patch
        depth_2d = padded_depth.reshape(patch_size, patch_size)
    else:
        depth_2d = depth_patch.reshape(patch_size, patch_size)

    # Apply Gaussian smoothing before gradient computation
    smoothed_depth = gaussian_filter(depth_2d, sigma=0.5)
    grad_y, grad_x = np.gradient(smoothed_depth)
    # Use nanmean to handle any NaN values and prevent "invalid value encountered" warnings
    gradient_magnitude = np.nanmean(np.sqrt(grad_x**2 + grad_y**2))
    # Replace NaN with 0 if all values were invalid
    if np.isnan(gradient_magnitude):
        gradient_magnitude = 0.0

    # Surface normal consistency (if surface normal is provided)
    surface_consistency = 0.0
    if surface_normal is not None:
        # Compute local surface normal from depth gradients
        local_normal = np.array([-np.mean(grad_x), -np.mean(grad_y), 1.0])
        local_normal = local_normal / np.linalg.norm(local_normal)

        # Consistency is dot product (higher = more consistent)
        surface_consistency = max(0.0, np.dot(surface_normal, local_normal))

    features = {
        "depth_texture_roughness": depth_roughness,
        "depth_texture_gradient": gradient_magnitude,
        "depth_texture_consistency": surface_consistency
    }

    # Normalize features if requested
    if config["normalize"]:
        features["depth_texture_roughness"] = np.clip(features["depth_texture_roughness"] / 0.01, 0, 1)
        features["depth_texture_gradient"] = np.clip(features["depth_texture_gradient"] / 0.005, 0, 1)

    return features
```