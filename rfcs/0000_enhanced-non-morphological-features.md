- Start Date: 2025-07-04
- RFC PR:

# RFC: Enhanced Non-Morphological Feature Extraction in Sensor Modules

## Summary

This RFC proposes enhancing non-morphological feature extraction in Monty's sensor modules by implementing multi-scale Local Binary Patterns (LBP) for texture analysis and enhanced depth processing for tactile-like features. The implementation uses a **pure composition approach** with a dedicated texture processor that can be integrated into any sensor module, improving object recognition and discrimination capabilities while maintaining rotation invariance as specified in the project documentation. This approach avoids multiple inheritance complexity and aligns with established architectural preferences in the codebase.

## Quick Start

**For users who want to try texture features immediately:**

1. **Add texture features to your experiment config:**
   ```python
   features = ["on_object", "hsv", "pose_vectors", "lbp_texture", "depth_roughness"]
   ```

2. **Run your experiment** - texture features will be automatically extracted with sensible defaults

3. **For better performance, add feature weights:**
   ```python
   feature_weights = {
       "patch": {
           "hsv": np.array([1, 0.5, 0.5]),
           "lbp_texture": np.ones(18) * 0.7,  # LBP histogram
           "depth_roughness": 1.0,
       }
   }
   ```

That's it! The system handles all the complexity internally while providing simple configuration options.

## Motivation

### Current Limitations

The current non-morphological feature extraction in sensor modules is limited to:
- **Single-pixel sampling**: Only extracts RGBA/HSV from the center pixel (1/4096 pixels used)
- **No texture information**: Cannot distinguish between smooth vs. textured surfaces with identical colors
- **Basic depth statistics**: Only min/mean depth, missing surface texture information
- **No rotation invariance**: Current features change with sensor orientation
- **Limited cross-modal integration**: RGB and depth features are processed independently

### Problem Statement

From the [Extract Better Features](https://thousandbrainsproject.readme.io/docs/extract-better-features) documentation:
> "Currently non-morphological features are very simple, such as extracting the RGB or hue value at the center of the sensor patch. In the short term, we would like to extract richer features, such as using HTM's spatial-pooler or Local Binary Patterns for visual features, or processing depth information within a patch to approximate tactile texture."

The documentation specifically requires **rotation invariant** features that can detect textured patterns regardless of sensor orientation.

### Research Questions

1. **Feature Complementarity**: Should LBP and HTM Spatial Pooler be used together for enhanced texture discrimination?
2. **Cross-Modal Integration**: How can RGB and depth texture features be optimally combined?
3. **Computational Trade-offs**: What is the acceptable computational overhead for improved recognition accuracy?
4. **Feature Interaction**: How do new texture features correlate with existing morphological features?

## Detailed Design

### Architecture Overview

The implementation follows a **pure composition approach** that can be integrated into any sensor module:

1. **Texture Processing Utilities**: Standalone functions in `src/tbp/monty/frameworks/utils/texture_processing.py`
2. **Texture Feature Processor**: Dedicated `TextureFeatureProcessor` class that handles all texture extraction logic
3. **Composition Integration**: Sensor modules use composition ("has-a" relationship) to incorporate texture processing
4. **Broad Applicability**: Not limited to Habitat-specific sensor modules, works with any RGBA+depth data
5. **Clean Architecture**: Avoids multiple inheritance complexity and method resolution order issues entirely
6. **Consistent with User Preferences**: Follows the established preference for composition over mixin patterns

### 1. Texture Processing Utilities

#### New File: `src/tbp/monty/frameworks/utils/texture_processing.py`

```python
"""Texture processing utilities for sensor modules."""

import numpy as np
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray, rgb2hsv
from scipy.ndimage import gaussian_filter


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


def extract_enhanced_depth_features(depth_patch, obs_3d, surface_normal=None, config=None):
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
            "depth_roughness": 0.0,
            "depth_gradient_magnitude": 0.0,
            "depth_surface_consistency": 0.0
        }

    on_object_depths = depth_patch[on_object_mask]

    # Handle edge case of insufficient data points
    if len(on_object_depths) < 2:
        return {
            "depth_roughness": 0.0,
            "depth_gradient_magnitude": 0.0,
            "depth_surface_consistency": 0.0
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
        "depth_roughness": depth_roughness,
        "depth_gradient_magnitude": gradient_magnitude,
        "depth_surface_consistency": surface_consistency
    }

    # Normalize features if requested
    if config["normalize"]:
        features["depth_roughness"] = np.clip(features["depth_roughness"] / 0.01, 0, 1)
        features["depth_gradient_magnitude"] = np.clip(features["depth_gradient_magnitude"] / 0.005, 0, 1)

    return features
```

#### Key Properties
- **Multi-scale Analysis**: LBP at different radii captures texture at various scales
- **Configurable**: Flexible parameter configuration for different use cases
- **Rotation Invariant**: LBP uses `method='uniform'` for rotation invariance
- **Channel Flexibility**: Can process grayscale, RGB, or individual color channels
- **Enhanced Depth Processing**: Surface normal consistency and improved gradient analysis
- **Composition-Based**: Clean architecture using "has-a" relationship instead of inheritance
- **Modular Design**: Functions can be used by any sensor module with RGBA+depth data
- **Testable**: Easy to unit test in isolation, no inheritance complexity
- **Non-Morphological**: All texture features are classified as non-morphological features in Monty's architecture

### 2. Texture Feature Processor

#### New File: `src/tbp/monty/frameworks/models/texture_processor.py`

```python
"""Texture feature processor for sensor modules using composition."""

import numpy as np
from tbp.monty.frameworks.utils.texture_processing import (
    extract_multiscale_lbp_features,
    extract_enhanced_depth_features,
)


class TextureFeatureProcessor:
    """Handles all texture feature extraction logic using composition.

    This class can be used by any sensor module that processes RGBA and depth data.
    It provides configurable texture feature extraction including multi-scale LBP
    and enhanced depth-based features.
    """

    def __init__(self, texture_config=None):
        """Initialize texture feature processor.

        Args:
            texture_config: Configuration dict for texture processing parameters.
                          Can include 'preset' key for simplified configuration.
        """
        # Handle preset configurations for simplified usage
        if texture_config and "preset" in texture_config:
            preset = texture_config["preset"]
            if preset == "basic":
                base_config = {
                    "lbp_params": {"radii": [2], "n_points": [16], "method": "uniform", "channels": "grayscale"},
                    "depth_params": {"roughness_scale": 1.0, "normalize": True}
                }
            elif preset == "enhanced":
                base_config = {
                    "lbp_params": {"radii": [1, 2], "n_points": [8, 16], "method": "uniform", "channels": "grayscale"},
                    "depth_params": {"roughness_scale": 1.0, "normalize": True, "gradient_kernel_size": 3}
                }
            else:  # complete
                base_config = {
                    "lbp_params": {"radii": [1, 2], "n_points": [8, 16], "method": "uniform", "channels": "grayscale"},
                    "depth_params": {"roughness_scale": 1.0, "normalize": True, "gradient_kernel_size": 3, "surface_consistency_threshold": 0.1}
                }
            # Merge with any additional config parameters
            self.config = {**base_config, **{k: v for k, v in texture_config.items() if k != "preset"}}
        else:
            # Default texture configuration (enhanced preset)
            self.config = texture_config or {
                "lbp_params": {
                    "radii": [1, 2],
                    "n_points": [8, 16],
                    "method": "uniform",
                    "channels": "grayscale"
                },
                "depth_params": {
                    "roughness_scale": 1.0,
                    "gradient_kernel_size": 3,
                    "normalize": True,
                    "surface_consistency_threshold": 0.1
                }
            }

        # Cache feature names for efficiency
        self._cached_feature_names = None

    def get_available_feature_names(self):
        """Get list of all possible texture feature names based on configuration."""
        if self._cached_feature_names is None:
            feature_names = []

            # LBP feature names
            lbp_config = self.config["lbp_params"]
            channels = [""] if lbp_config["channels"] == "grayscale" else ["_r", "_g", "_b"]

            for ch_name in channels:
                for radius, n_points in zip(lbp_config["radii"], lbp_config["n_points"]):
                    feature_names.append(f"lbp_texture{ch_name}_r{radius}_p{n_points}")

            # Depth feature names
            feature_names.extend([
                "depth_roughness",
                "depth_gradient_magnitude",
                "depth_surface_consistency"
            ])

            self._cached_feature_names = feature_names

        return self._cached_feature_names

    def extract_features(self, rgba_feat, depth_feat, obs_3d, requested_features, surface_normal=None):
        """Extract requested texture features from RGBA and depth data.

        Args:
            rgba_feat: RGBA patch data (H x W x 4)
            depth_feat: Depth patch data (flattened)
            obs_3d: 3D observation data with object mask (N x 4)
            requested_features: List of feature names to extract
            surface_normal: Optional surface normal vector

        Returns:
            dict: Extracted texture features (only requested ones)
        """
        texture_features = {}

        # Validate inputs to prevent numpy warnings
        if rgba_feat is None or rgba_feat.size == 0:
            return texture_features
        if depth_feat is None or depth_feat.size == 0:
            return texture_features
        if obs_3d is None or obs_3d.size == 0:
            return texture_features

        # Extract LBP features if any are requested
        lbp_feature_names = [f for f in requested_features if f.startswith("lbp_texture")]
        if lbp_feature_names:
            try:
                lbp_features = extract_multiscale_lbp_features(
                    rgba_feat, self.config["lbp_params"]
                )
                # Only include requested LBP features
                for feature_name in lbp_feature_names:
                    if feature_name in lbp_features:
                        texture_features[feature_name] = lbp_features[feature_name]
            except Exception as e:
                # Log warning but continue processing other features
                import logging
                logging.getLogger(__name__).warning(f"LBP feature extraction failed: {e}")

        # Extract depth features if any are requested
        depth_feature_names = [f for f in requested_features if f.startswith("depth_")]
        if depth_feature_names:
            try:
                depth_features = extract_enhanced_depth_features(
                    depth_feat, obs_3d, surface_normal, self.config["depth_params"]
                )
                # Only include requested depth features
                for feature_name in depth_feature_names:
                    if feature_name in depth_features:
                        texture_features[feature_name] = depth_features[feature_name]
            except Exception as e:
                # Log warning but continue processing other features
                import logging
                logging.getLogger(__name__).warning(f"Depth feature extraction failed: {e}")

        return texture_features

    def has_texture_features(self, feature_list):
        """Check if any texture features are requested in the feature list."""
        available_features = self.get_available_feature_names()
        return any(feature in available_features for feature in feature_list)

    def update_config(self, new_config):
        """Update texture processing configuration and clear cache."""
        self.config.update(new_config)
        self._cached_feature_names = None  # Clear cache to regenerate feature names

    def get_noise_compatible_features(self):
        """Get texture features that are compatible with the existing noise system."""
        # LBP histograms are arrays that can have gaussian noise added
        # Depth features are scalars that can have gaussian noise added
        return {
            "lbp_texture": "array",  # histogram arrays
            "depth_roughness": "scalar",
            "depth_gradient_magnitude": "scalar",
            "depth_surface_consistency": "scalar"
        }
```

### 3. Integration with Existing Sensor Modules

#### Enhanced HabitatDistantPatchSM with Texture Features

```python
# In src/tbp/monty/frameworks/models/sensor_modules.py
from tbp.monty.frameworks.models.texture_processor import TextureFeatureProcessor

class HabitatDistantPatchSM(DetailedLoggingSM, NoiseMixin):
    """Enhanced Habitat sensor module with texture feature extraction capabilities."""

    def __init__(self, sensor_module_id, features, save_raw_obs=False,
                 pc1_is_pc2_threshold=10, noise_params=None, process_all_obs=False,
                 texture_config=None):
        # Initialize parent classes using existing pattern
        super(HabitatDistantPatchSM, self).__init__(
            sensor_module_id, save_raw_obs, pc1_is_pc2_threshold
        )
        NoiseMixin.__init__(self, noise_params)

        # Initialize existing attributes to match current implementation
        self.features = features
        self.processed_obs = []
        self.states = []
        self.on_object_obs_only = True
        self.process_all_obs = process_all_obs

        # Create texture processor using composition if texture features are requested
        self.texture_processor = None
        if self._has_texture_features(features):
            self.texture_processor = TextureFeatureProcessor(texture_config)

        # Use existing possible_features list and extend it
        possible_features = [
            "on_object", "object_coverage", "min_depth", "mean_depth",
            "rgba", "hsv", "pose_vectors", "principal_curvatures",
            "principal_curvatures_log", "pose_fully_defined",
            "gaussian_curvature", "mean_curvature", "gaussian_curvature_sc",
            "mean_curvature_sc", "curvature_for_TM", "coords_for_TM",
        ]

        # Add texture features to possible features if processor exists
        if self.texture_processor:
            texture_features = self.texture_processor.get_available_feature_names()
            possible_features.extend(texture_features)

        # Validate requested features against possible features
        for feature in features:
            assert feature in possible_features, (
                f"{feature} not part of {possible_features}"
            )

    def _has_texture_features(self, feature_list):
        """Check if any texture features are requested."""
        return any(f.startswith("lbp_texture") or f.startswith("depth_")
                  for f in feature_list)



    def extract_and_add_features(self, features, obs_3d, rgba_feat, depth_feat,
                                center_id, center_row_col, sensor_frame_data, world_camera):
        """Enhanced feature extraction including texture features."""

        # Call parent method for existing features
        features, morphological_features, invalid_signals = super().extract_and_add_features(
            features, obs_3d, rgba_feat, depth_feat, center_id, center_row_col,
            sensor_frame_data, world_camera
        )

        # Extract texture features if processor exists and features are requested
        if self.texture_processor:
            texture_feature_names = [f for f in self.features if
                                   f.startswith("lbp_texture") or f.startswith("depth_")]

            if texture_feature_names:
                # Get surface normal for enhanced depth processing (handle missing pose_vectors)
                surface_normal = None
                if "pose_vectors" in morphological_features and morphological_features["pose_vectors"]:
                    surface_normal = morphological_features["pose_vectors"][0]

                # Extract texture features using composition
                texture_features = self.texture_processor.extract_features(
                    rgba_feat, depth_feat, obs_3d, texture_feature_names, surface_normal
                )

                # Add texture features to features dict (they are non-morphological)
                features.update(texture_features)

        return features, morphological_features, invalid_signals

    def pre_episode(self):
        """Reset buffer and is_exploring flag."""
        super().pre_episode()
        self.processed_obs = []
        self.states = []

    def update_state(self, state):
        """Update information about the sensors location and rotation."""
        # Use existing implementation from parent class
        super().update_state(state)

    def step(self, data):
        """Turn raw observations into dict of features at location."""
        # Use existing implementation from parent class
        return super().step(data)
```

### 4. Learning Module Integration

#### Enhanced Feature Weights and Tolerances
```python
# In benchmarks/configs/defaults.py or experiment configs
enhanced_texture_feature_weights = {
    "patch": {
        # Existing features
        "hsv": np.array([1, 0.5, 0.5]),

        # Multi-scale LBP features (weights match histogram bin counts)
        "lbp_texture_r1_p8": np.ones(10) * 0.8,    # Uniform LBP: 8+2=10 bins
        "lbp_texture_r2_p16": np.ones(18) * 0.7,   # Uniform LBP: 16+2=18 bins

        # Enhanced depth features
        "depth_roughness": 1.0,
        "depth_gradient_magnitude": 1.0,
        "depth_surface_consistency": 0.5,  # Lower weight for consistency check
    }
}

# Enhanced tolerance values with multi-scale support
enhanced_tolerance_values = {
    # Existing features
    "hsv": np.array([0.1, 0.2, 0.2]),

    # Multi-scale LBP tolerances (tighter for fine-scale, looser for coarse-scale)
    "lbp_texture_r1_p8": np.ones(10) * 0.12,     # Fine texture, tighter tolerance
    "lbp_texture_r2_p16": np.ones(18) * 0.18,    # Coarse texture, looser tolerance

    # Enhanced depth feature tolerances
    "depth_roughness": 0.01,                      # 1cm roughness tolerance
    "depth_gradient_magnitude": 0.005,            # 5mm gradient tolerance
    "depth_surface_consistency": 0.15,            # 15% consistency tolerance
}

# Simplified configuration presets for easy adoption
texture_feature_configs = {
    "basic": {
        "features": ["lbp_texture", "depth_roughness"],
        "description": "Basic texture analysis - single scale LBP + depth roughness"
    },
    "enhanced": {
        "features": ["lbp_texture", "lbp_texture_fine", "depth_roughness", "depth_gradient"],
        "description": "Enhanced texture analysis - multi-scale LBP + enhanced depth"
    },
    "complete": {
        "features": ["lbp_texture", "lbp_texture_fine", "depth_roughness",
                    "depth_gradient", "depth_consistency"],
        "description": "Complete texture feature set with all available features"
    }
}

# Simplified feature names (automatically mapped to full names internally)
simplified_feature_mapping = {
    "lbp_texture": "lbp_texture_r2_p16",      # Default: coarse scale
    "lbp_texture_fine": "lbp_texture_r1_p8",  # Fine scale option
    "depth_roughness": "depth_roughness",
    "depth_gradient": "depth_gradient_magnitude",
    "depth_consistency": "depth_surface_consistency"
}
```

#### Simplified Usage Examples

**Basic Usage (Minimal Configuration):**
```python
# Just add texture features to existing feature list
features = ["on_object", "hsv", "pose_vectors", "lbp_texture", "depth_roughness"]

sensor_module = HabitatDistantPatchSM(
    sensor_module_id="patch",
    features=features,
    # texture_config defaults to sensible values
)
```

**Advanced Usage (Custom Configuration):**
```python
# Use preset configuration
texture_config = texture_feature_configs["enhanced"]

sensor_module = HabitatDistantPatchSM(
    sensor_module_id="patch",
    features=texture_config["features"] + ["on_object", "hsv", "pose_vectors"],
    texture_config={
        "preset": "enhanced",  # or "basic", "complete"
        "lbp_channels": "grayscale"  # or "rgb" for color texture
    }
)
```

#### Backward Compatibility and Migration
- **Existing configurations continue to work** - new features are opt-in
- **Default behavior unchanged** - only extract new features if explicitly requested
- **Gradual adoption** - texture feature configs allow incremental feature addition
- **Feature weights can be zero** - effectively disable features if problematic
- **Configuration validation** - warn users about unknown texture feature names
- **Simplified feature names** - users can use "lbp_texture" instead of "lbp_texture_r2_p16"

#### Integration with Existing Noise System
- **Texture features support noise** - compatible with existing `NoiseMixin` architecture
- **Appropriate noise types** - LBP histograms get gaussian noise, depth features get scalar noise
- **Noise configuration example**:
```python
texture_noise_params = {
    "features": {
        "lbp_texture_r1_p8": 0.05,  # 5% noise on histogram values
        "depth_roughness": 0.001,   # 1mm noise on roughness
        "depth_gradient_magnitude": 0.0005,  # 0.5mm noise on gradients
    }
}
```

## Implementation Plan

### Phase 1: Core Infrastructure
1. **Create texture processing utilities** in `src/tbp/monty/frameworks/utils/texture_processing.py`
   - `extract_multiscale_lbp_features()` function with configurable parameters
   - `extract_enhanced_depth_features()` function with surface normal integration
   - Comprehensive docstrings, type hints, and parameter validation
2. **Create TextureFeatureProcessor** in `src/tbp/monty/frameworks/models/texture_processor.py`
   - Standalone processor class using composition pattern
   - Configurable texture processing parameters
   - Automatic feature name generation and validation
   - Clean interface for integration with any sensor module
3. **Unit tests** for utility functions in `tests/unit/texture_processing_test.py`
   - Test multi-scale LBP extraction
   - Test depth feature computation
   - Test rotation invariance properties
   - Test edge cases and error handling
4. **Unit tests** for processor class in `tests/unit/texture_processor_test.py`
   - Test processor initialization and configuration
   - Test feature extraction interface
   - Test feature name generation
   - Test configuration updates

### Phase 2: Sensor Module Integration
1. **Enhance HabitatDistantPatchSM** with TextureFeatureProcessor composition
   - Clean composition integration using "has-a" relationship
   - Enhanced `extract_and_add_features()` method
   - Surface normal integration for depth features
   - Follow existing patterns from NoiseMixin integration
   - No multiple inheritance complexity
2. **Create example sensor modules** demonstrating composition usage
   - Basic texture-enabled sensor module
   - Advanced multi-scale texture sensor module
   - Custom sensor modules with texture processing
3. **Integration tests** with existing sensor module hierarchy
   - Test composition integration compatibility
   - Test feature extraction pipeline integration
   - Test processor lifecycle management
   - Test integration with existing noise system

### Phase 3: Configuration and Validation
1. **Enhanced configuration system** in `benchmarks/configs/defaults.py`
   - Multi-scale feature weights and tolerances
   - Texture feature configuration presets
   - Migration helpers for existing configurations
2. **Comprehensive validation experiments**
   - Controlled texture discrimination tests
   - Rotation invariance validation
   - Multi-scale analysis effectiveness
   - Computational performance benchmarking
3. **Benchmark integration** with existing YCB experiments
   - Baseline comparison experiments
   - Performance regression testing
   - Feature contribution analysis

### Phase 4: Advanced Features and Optimization
1. **Performance optimization**
   - Parallel processing for multi-scale LBP
   - Memory usage optimization for large patches
   - Computational efficiency improvements
2. **Advanced texture features**
   - Cross-modal RGB-depth texture integration
   - Adaptive feature selection based on object type
   - HTM Spatial Pooler integration (future work)
3. **Documentation and examples**
   - Usage tutorials and best practices
   - Performance tuning guidelines
   - Integration examples for custom sensor modules

## Evaluation Criteria

### Success Metrics
1. **Texture Discrimination**: Ability to distinguish objects with similar colors but different textures
   - Target: >90% accuracy on controlled texture discrimination tasks
   - Baseline: Current HSV-only performance on similar objects
2. **Rotation Invariance**: Multi-scale LBP features remain consistent across sensor orientations
   - Target: <5% feature variation across 0°-360° rotations
   - Metric: Feature vector cosine similarity across rotations
3. **Recognition Accuracy**: Improved object recognition performance on existing benchmarks
   - Target: ≥5% improvement on YCB 10-object recognition task
   - Baseline: Current best-performing configuration without texture features
4. **Computational Efficiency**: Feature extraction time remains reasonable
   - Target: <15ms per patch for full multi-scale texture extraction
   - Baseline: Current feature extraction time (~3-5ms per patch)
5. **Multi-scale Effectiveness**: Different texture scales provide complementary information
   - Target: Multi-scale features outperform single-scale by ≥3%
   - Metric: Ablation study comparing single vs. multi-scale configurations

### Comprehensive Validation Framework

#### 1. Controlled Texture Experiments
```python
texture_validation_experiments = {
    "rotation_invariance": {
        "objects": ["textured_sphere", "ribbed_cylinder", "woven_cube"],
        "rotations": [0, 45, 90, 135, 180, 225, 270, 315],  # degrees
        "metrics": ["feature_consistency", "recognition_stability"],
        "success_criteria": "cosine_similarity > 0.95 across rotations"
    },
    "texture_discrimination": {
        "object_pairs": [
            ("smooth_red_sphere", "textured_red_sphere"),
            ("plain_blue_cube", "ribbed_blue_cube"),
            ("flat_green_cylinder", "grooved_green_cylinder")
        ],
        "metrics": ["classification_accuracy", "confusion_matrix"],
        "success_criteria": "accuracy > 90% for same-color different-texture pairs"
    },
    "multi_scale_analysis": {
        "textures": ["fine_grain", "medium_grain", "coarse_grain", "mixed_scale"],
        "lbp_configs": [
            {"radii": [1], "n_points": [8]},      # Fine only
            {"radii": [2], "n_points": [16]},     # Coarse only
            {"radii": [1, 2], "n_points": [8, 16]}  # Multi-scale
        ],
        "success_criteria": "multi_scale accuracy > single_scale + 3%"
    }
}
```

#### 2. Benchmark Integration Tests
- **YCB 10-object recognition**: Compare texture-enhanced vs. baseline performance
- **Noisy conditions**: Validate robustness under various noise conditions
- **Multi-object scenes**: Test texture features in complex environments
- **Real-world validation**: iPad camera experiments with texture-rich objects

#### 3. Performance Analysis
- **Feature extraction timing**: Measure computational overhead per texture feature type
- **Memory usage**: Track memory consumption for multi-scale feature storage
- **Scalability**: Test performance with larger patch sizes and more texture scales
- **Feature correlation**: Analyze correlation between texture and morphological features

## Alternatives Considered

### HTM Spatial Pooler
- **Pros**: Aligns with TBT principles, adaptive learning, sparse distributed representations
- **Cons**: High implementation complexity, no inherent rotation invariance, requires training
- **Decision**: Implement as complementary approach in Phase 4, not replacement for LBP
- **Future Integration**: Could be used alongside LBP for learned texture representations

### Gabor Filters
- **Pros**: Multi-scale texture analysis, biologically inspired, frequency domain analysis
- **Cons**: Not rotation invariant, high dimensionality, parameter sensitivity
- **Decision**: LBP provides better rotation invariance with lower computational cost
- **Potential Use**: Consider for specialized texture analysis in future work

### Advanced Color Spaces (LAB, XYZ, etc.)
- **Pros**: Better perceptual uniformity than HSV, device-independent color representation
- **Cons**: Still lighting dependent, limited texture discrimination power, conversion overhead
- **Decision**: Keep existing HSV, focus computational resources on texture improvements
- **Future Consideration**: Evaluate for lighting-invariant color features

### Deep Learning Feature Extractors
- **Pros**: State-of-the-art texture recognition, learned representations, transfer learning
- **Cons**: Conflicts with TBP philosophy, high computational cost, black-box features
- **Decision**: Explicitly avoided per project FAQ on deep learning usage
- **Alternative**: Investigate shallow CNN alternatives in future work

### Wavelet-Based Texture Analysis
- **Pros**: Multi-resolution analysis, good for texture characterization, frequency domain
- **Cons**: Not inherently rotation invariant, complex parameter tuning, higher computational cost
- **Decision**: LBP provides simpler, more robust rotation-invariant alternative
- **Consideration**: Could complement LBP for specific texture types

### Integration Architecture Alternatives

#### Direct Integration vs. Composition Approach
- **Direct Integration**: Modify each sensor module individually
  - **Pros**: Simpler initial implementation, sensor-specific optimizations
  - **Cons**: Code duplication, maintenance burden, limited reusability
- **Composition Approach** (Selected): Reusable texture feature processor
  - **Pros**: Code reuse, consistent implementation, easy adoption, modular design, no inheritance complexity
  - **Cons**: Slightly more complex initialization, requires careful lifecycle management
- **Decision**: Composition approach provides better long-term maintainability and cleaner architecture

#### Feature Processing Location
- **Sensor Module Level** (Selected): Process features during sensor observation
  - **Pros**: Consistent with existing architecture, real-time processing, sensor-specific optimizations
  - **Cons**: Computational overhead during observation, memory usage
- **Learning Module Level**: Process features during learning/inference
  - **Pros**: Reduced sensor module complexity, batch processing opportunities
  - **Cons**: Architectural inconsistency, delayed feature availability
- **Decision**: Sensor module level maintains architectural consistency

## Risks and Mitigations

### Risk 1: Performance Regression
- **Risk**: New texture features might hurt performance on objects where texture is not discriminative
- **Mitigation**:
  - Hybrid approach maintains existing HSV features as baseline
  - Feature weights can be set to zero to disable problematic features
  - Gradual rollout with A/B testing on different object types
- **Monitoring**: Track performance metrics per object type and feature combination
- **Fallback**: Configuration presets allow quick reversion to baseline features

### Risk 2: Computational Overhead
- **Risk**: Multi-scale LBP and enhanced depth processing may significantly increase processing time
- **Mitigation**:
  - LBP is computationally efficient (simple pixel comparisons and histograms)
  - Configurable texture parameters allow performance tuning
  - Parallel processing opportunities for multi-scale analysis
- **Monitoring**: Continuous benchmarking of feature extraction timing
- **Optimization**: Profile-guided optimization and caching strategies

### Risk 3: Integration Complexity
- **Risk**: Composition approach may require careful lifecycle management of texture processor
- **Mitigation**:
  - Clear initialization and cleanup patterns for texture processor
  - Comprehensive testing of processor lifecycle
  - Well-defined interface between sensor modules and processor
  - Follow existing patterns from NoiseMixin integration
- **Testing**: Extensive integration tests with different sensor module combinations
- **Benefits**: Composition avoids multiple inheritance complexity entirely and aligns with user preferences

### Risk 4: Feature Dimensionality Explosion
- **Risk**: Multi-scale LBP features may create very high-dimensional feature vectors
- **Mitigation**:
  - Configurable feature selection (choose specific scales/channels)
  - Feature importance analysis to identify most discriminative components
  - Dimensionality reduction techniques if needed
- **Monitoring**: Track memory usage and learning module performance with high-dimensional features

### Risk 5: Rotation Invariance Limitations
- **Risk**: LBP uniform patterns may not capture all texture variations under rotation
- **Mitigation**:
  - Comprehensive rotation invariance testing across diverse textures
  - Fallback to rotation-variant LBP if uniform patterns prove insufficient
  - Combination with other rotation-invariant features
- **Validation**: Quantitative rotation invariance metrics in evaluation framework

### Risk 6: Configuration Complexity
- **Risk**: Many texture parameters may make system difficult to configure and tune
- **Mitigation**:
  - **Simplified feature names**: Users can specify "lbp_texture" instead of "lbp_texture_r2_p16"
  - **Smart defaults**: Sensible default configurations work out-of-the-box
  - **Configuration presets**: "basic", "enhanced", "complete" presets for different needs
  - **Automatic parameter mapping**: Complex internal parameters hidden from users
  - **Progressive complexity**: Start simple, add complexity only when needed
- **Documentation**: Clear usage examples and parameter tuning guides

### Risk 7: Numpy Runtime Warnings
- **Risk**: Empty patches or invalid depth data may cause numpy warnings like "Mean of empty slice" or "invalid value encountered in double_scalars"
- **Mitigation**:
  - Explicit validation of input data before processing
  - Proper handling of edge cases with empty or invalid patches
  - Use of numpy functions with appropriate handling of NaN/empty values
- **Implementation**: Add robust error checking in texture processing functions

## Dependencies

### External Libraries
- **scikit-image**: Already available in pyproject.toml, provides `local_binary_pattern` and `rgb2gray`
- **numpy**: Already available, for histogram computation and gradient operations

### Internal Dependencies
- **Existing sensor module architecture**: Integrates with `HabitatDistantPatchSM`
- **Existing feature processing pipeline**: Uses current `extract_and_add_features()` pattern
- **No breaking changes**: Maintains backward compatibility with existing configurations

### Enhanced File Structure
```
src/tbp/monty/frameworks/utils/
├── sensor_processing.py          # existing
├── texture_processing.py         # NEW - multi-scale texture utility functions
└── ...

src/tbp/monty/frameworks/models/
├── sensor_modules.py             # MODIFIED - enhanced with TextureFeatureProcessor composition
├── texture_processor.py          # NEW - TextureFeatureProcessor class
└── ...

tests/unit/
├── sensor_module_test.py         # existing
├── texture_processing_test.py    # NEW - utility function tests
├── texture_processor_test.py     # NEW - processor class tests
└── ...

tests/integration/
├── texture_feature_integration_test.py  # NEW - end-to-end texture feature tests
└── ...

benchmarks/configs/
├── defaults.py                   # MODIFIED - enhanced texture feature configs
├── texture_experiments.py       # NEW - texture-specific experiment configs
└── ...
```

## Future Work

### Phase 5: Advanced Texture Analysis
1. **HTM Spatial Pooler Integration**: Implement as complementary learned texture representation
   - Sparse distributed representations for texture patterns
   - Online learning of texture vocabularies
   - Integration with existing LBP features for hybrid approach
2. **Cross-Modal Texture Fusion**: Advanced RGB-depth texture integration
   - Joint RGB-depth texture descriptors
   - Cross-modal texture consistency validation
   - Multi-modal texture similarity metrics
3. **Adaptive Multi-Scale Analysis**: Dynamic scale selection based on texture characteristics
   - Automatic scale parameter tuning per object type
   - Texture complexity-based feature selection
   - Computational resource-aware scale adaptation

### Research Directions
1. **Learned Texture Representations**: Investigate shallow CNN alternatives within TBP constraints
   - Biologically-inspired convolutional layers
   - Unsupervised texture feature learning
   - Transfer learning from texture databases
2. **Temporal Texture Analysis**: Texture features across sensorimotor sequences
   - Texture consistency across viewpoints
   - Temporal texture pattern recognition
   - Motion-based texture enhancement
3. **Hierarchical Texture Processing**: Multi-level texture analysis in cortical hierarchy
   - Texture feature abstraction across learning module levels
   - Compositional texture representations
   - Texture-based object part decomposition

### Long-Term Vision
1. **Universal Texture Understanding**: Texture features that work across all sensory modalities
   - Tactile texture from force/pressure sensors
   - Auditory texture from sound patterns
   - Cross-modal texture transfer and generalization
2. **Neuromorphic Texture Processing**: Hardware-accelerated texture feature extraction
   - Spike-based LBP computation
   - Neuromorphic depth processing
   - Real-time texture analysis on edge devices
3. **Texture-Guided Active Sensing**: Use texture features to guide exploration policies
   - Texture-based attention mechanisms
   - Texture-informed saccade planning
   - Texture-driven hypothesis testing

## Revision Notes

### Issues Addressed in This RFC

1. **Import Statement Correction**: Added missing `rgb2hsv` import from `skimage.color`
2. **Constructor Parameter Alignment**: Ensured `HabitatDistantPatchSM` constructor matches existing implementation with `process_all_obs` parameter
3. **Attribute Initialization**: Added proper initialization of existing attributes (`processed_obs`, `states`, `on_object_obs_only`, `process_all_obs`)
4. **Method Completeness**: Added `pre_episode`, `update_state`, and `step` method implementations to maintain compatibility
5. **Feature Validation**: Used existing assertion pattern for feature validation instead of custom validation method
6. **Architecture Consistency**: Ensured texture features are properly categorized as non-morphological features
7. **Error Handling**: Improved robustness for edge cases with empty patches and invalid data
8. **Numpy Warnings Prevention**: Added proper handling to prevent "Mean of empty slice" and "invalid value encountered in double_scalars" warnings

### Key Design Decisions

- **Composition over Inheritance**: TextureFeatureProcessor uses composition to avoid multiple inheritance complexity
- **Backward Compatibility**: All existing functionality is preserved, new features are opt-in
- **Feature Classification**: Texture features are correctly classified as non-morphological
- **Configuration Flexibility**: Multiple configuration presets for different use cases
- **Performance Considerations**: Lazy initialization of texture processor only when needed

## Questions for Community

### Technical Design Questions
1. **Composition Architecture**: Is the TextureFeatureProcessor composition approach optimal?
   - Should the processor be created lazily only when texture features are needed?
   - How should we handle processor lifecycle (creation, updates, cleanup)?
   - Does this approach align well with the existing NoiseMixin integration pattern?

2. **Feature Naming Convention**: Are the proposed multi-scale feature names intuitive?
   - `lbp_texture_r1_p8` for radius=1, points=8 LBP features
   - `depth_surface_consistency` for surface normal consistency checking
   - Should we use shorter names or more descriptive ones?

3. **Configuration Complexity**: How should we balance flexibility vs. simplicity?
   - Are the texture configuration presets (basic/multiscale/full) sufficient?
   - Should advanced parameters be hidden from typical users?

4. **Error Handling**: How should we handle edge cases in texture processing?
   - Empty patches or patches with insufficient on-object points
   - Invalid depth data or RGBA patches
   - Should we fail silently, log warnings, or raise exceptions?

5. **Noise Integration**: How should texture features integrate with the existing noise system?
   - Are the proposed noise types (gaussian for histograms/scalars) appropriate?
   - Should texture features have different noise characteristics than morphological features?

### Integration and Adoption Questions
6. **Default Behavior**: Should new texture features be:
   - Completely opt-in (current proposal)
   - Gradually enabled by default in new experiments
   - Enabled by default with easy disable options

7. **Backward Compatibility**: How strict should we be about maintaining existing behavior?
   - Should we provide automatic migration tools for old configs?
   - Is it acceptable to change default feature sets in major versions?

8. **Performance Requirements**: What are acceptable computational overhead limits?
   - Target: <15ms per patch for full multi-scale extraction (vs. ~5ms current)
   - Should we provide performance vs. accuracy trade-off options?

### Validation and Benchmarking Questions
9. **Benchmark Priority**: Which experiments should be prioritized for validation?
   - YCB 10-object recognition (standard benchmark)
   - Texture-specific discrimination tasks (new experiments)
   - Real-world iPad camera experiments
   - Multi-object scene understanding

10. **Success Criteria**: Are the proposed success metrics appropriate?
   - ≥5% improvement on YCB recognition
   - >90% accuracy on texture discrimination
   - <5% feature variation across rotations

### Research Direction Questions
11. **HTM Spatial Pooler Integration**: Should HTM SP be:
   - Implemented in parallel with LBP (complementary approach)
   - Considered as an alternative to LBP (replacement approach)
   - Deferred until LBP is fully validated

12. **Cross-Modal Integration**: How important is RGB-depth texture fusion?
    - Should it be part of the initial implementation?
    - Is the current separate processing approach sufficient?

### Community Feedback Requests
13. **Use Cases**: What specific texture-based recognition challenges are community members facing?
14. **Parameter Tuning**: What tools or guidelines would help with texture feature configuration?
15. **Documentation Needs**: What examples and tutorials would be most valuable?
16. **Testing Support**: What texture datasets or test objects would be useful for validation?

## References

### Primary Documentation
1. [Extract Better Features Documentation](https://thousandbrainsproject.readme.io/docs/extract-better-features) - Original task specification
2. [Thousand Brains Project FAQ](https://thousandbrainsproject.readme.io/docs/faq-monty#why-does-monty-not-make-use-of-deep-learning) - Deep learning usage guidelines
3. Current sensor module implementation: `src/tbp/monty/frameworks/models/sensor_modules.py`
4. Current feature processing: `src/tbp/monty/frameworks/utils/sensor_processing.py`

### Technical References
5. Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). "Multiresolution gray-scale and rotation invariant texture classification with local binary patterns." IEEE Transactions on Pattern Analysis and Machine Intelligence, 24(7), 971-987.
6. Ojala, T., Pietikäinen, M., & Harwood, D. (1996). "A comparative study of texture measures with classification based on featured distributions." Pattern Recognition, 29(1), 51-59.
7. Liao, S., Law, M. W., & Chung, A. C. (2009). "Dominant local binary patterns for texture classification." IEEE Transactions on Image Processing, 18(5), 1107-1118.

### Implementation References
8. scikit-image Local Binary Pattern implementation: https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.local_binary_pattern
9. Current benchmarking framework: `benchmarks/configs/defaults.py`
10. Evidence-based learning module: `src/tbp/monty/frameworks/models/evidence_matching/`

### Related Work in TBP
11. PR #343 - Reference standard for experimental analysis format and documentation depth
12. Existing noise handling in sensor modules: `src/tbp/monty/frameworks/models/mixins/noise.py`
13. Current feature weight and tolerance configurations: `benchmarks/configs/ycb_experiments.py`
