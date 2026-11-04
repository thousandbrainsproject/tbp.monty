- Start Date: 2025-07-06
- RFC PR: (leave this empty, it will be filled in after RFC is merged)

# Summary

We propose implementing a clean, interpretable CMP (Cortical Messaging Protocol) interpolation module that addresses current limitations in displacement-based communication between cortical columns. The module will use quaternion SLERP for orientation interpolation and geometry-based sinusoidal warping for 3D displacement interpolation, avoiding machine learning approaches to maintain biological interpretability.

# Motivation

## Current Problem

The existing CMP implementation suffers from artifacts when performing displacement-based interpolation due to linear averaging approaches. As documented in the [CMP displacement future work](../docs/future-work/voting-improvements/can-we-change-the-cmp-to-use-displacements-instead-of-locations.md), movement is core to how Learning Modules (LMs) process and model the world, yet the current system:

1. **Relies on body-centric locations** as primary spatial information, requiring LMs to infer displacements in object-centric coordinates
2. **Uses linear interpolation** for displacement calculations, which introduces geometric distortions
3. **Lacks proper orientation handling** during displacement transitions between poses

## Use Cases This Supports

- **Improved voting between LMs**: More accurate displacement-based communication when multiple sensors touch different parts of an object
- **Better goal-state transitions**: Smoother interpolation when transitioning between target poses
- **Enhanced flow detection**: Supporting future work on local and global flow detection
- **Moving object modeling**: Foundation for handling dynamic objects in the environment

## Expected Outcomes

- Elimination of geometric artifacts in displacement interpolation
- More biologically plausible spatial processing aligned with cortical column theory
- Improved accuracy in multi-sensor voting scenarios
- Foundation for transitioning CMP to displacement-primary communication

# Guide-level explanation

## Core Concept

Instead of linearly averaging displacements and orientations, the proposed system treats spatial transitions as movements along natural geometric paths:

1. **Orientations follow spherical paths** using quaternion SLERP (Spherical Linear Interpolation)
2. **Displacements follow sinusoidal curves** that respect the underlying geometry of sensor movement

## Example Usage

Consider two sensors touching a coffee mug - one sensing the rim, another the handle. When they vote on object identity:

**Current approach:**
```python
# Linear interpolation causes distortion
displacement_vote = (displacement_1 + displacement_2) / 2
orientation_vote = (orientation_1 + orientation_2) / 2  # Invalid for rotations!
```

**Proposed approach:**
```python
# Geometric interpolation preserves spatial relationships
interpolator = CMPDisplacementInterpolator()
result = interpolator.interpolate(
    pose_1=(displacement_1, quaternion_1, angle_1),
    pose_2=(displacement_2, quaternion_2, angle_2),
    target_angle=target_angle
)
displacement_vote, orientation_vote = result
```

### Concrete Numerical Example

To illustrate the geometric distortion problem, consider two sensors at 45° and 135° angles around a cylindrical object:

**Sensor 1 (45°):**
- Displacement: `[0.707, 0.707, 0.0]` (normalized)
- Orientation: `[0.0, 0.0, 0.383, 0.924]` (quaternion for 45° Z-rotation)

**Sensor 2 (135°):**
- Displacement: `[-0.707, 0.707, 0.0]` (normalized)
- Orientation: `[0.0, 0.0, 0.924, 0.383]` (quaternion for 135° Z-rotation)

**Linear averaging (current):**
- Displacement: `[0.0, 0.707, 0.0]` → Points at 90°, magnitude 0.707
- Orientation: `[0.0, 0.0, 0.654, 0.654]` → Invalid quaternion (not normalized)

**Geometric interpolation (proposed):**
- Target angle: 90°
- Displacement: `[0.0, 1.0, 0.0]` → Correct direction, proper magnitude
- Orientation: `[0.0, 0.0, 0.707, 0.707]` → Valid normalized quaternion for 90°

The linear approach produces a shortened displacement vector and an invalid orientation, while geometric interpolation maintains proper spatial relationships.

## Impact on Developers

- **Sensor Modules**: Will output quaternion orientations alongside displacement vectors
- **Learning Modules**: Will receive geometrically accurate interpolated poses during voting
- **Motor Systems**: Will benefit from smoother goal-state transitions
- **Existing Code**: Minimal changes required due to clean API design

# Reference-level explanation

## API Design

### Input Format
```python
@dataclass
class PoseWithAngle:
    displacement: np.ndarray  # 3D vector
    orientation: Quaternion   # scipy.spatial.transform.Rotation as quaternion
    angle: float             # Parameter for interpolation (e.g., sensor angle)
```

### Core Interface
```python
class CMPDisplacementInterpolator:
    def interpolate(
        self, 
        pose_1: PoseWithAngle, 
        pose_2: PoseWithAngle, 
        target_angle: float
    ) -> Tuple[np.ndarray, Quaternion]:
        """
        Interpolate between two poses using geometric methods.
        
        Returns:
            (interpolated_displacement, interpolated_orientation)
        """
```

## Algorithm Steps

1. **Compute interpolation parameter**: `t = (target_angle - angle_1) / (angle_2 - angle_1)`

2. **Orientation interpolation**: `orientation_result = SLERP(orientation_1, orientation_2, t)`
   - **Antipodal quaternion handling**: If `dot(q1, q2) < 0`, negate one quaternion to ensure shortest path
   - **Fallback**: If quaternions are nearly antipodal (`|dot(q1, q2)| < 0.01`), use linear interpolation with normalization
   - **Normalization**: Renormalize result and log warnings if drift >1e-6

3. **Displacement interpolation**: For each axis i ∈ {x, y, z}:
   - Solve linear system: `[sin(angle_1), cos(angle_1); sin(angle_2), cos(angle_2)] * [A_i; B_i] = [disp_1_i; disp_2_i]`
   - **Singular matrix handling**: If `det(matrix) < 1e-10`, fall back to linear interpolation for that axis
   - Evaluate: `disp_result_i = A_i * sin(target_angle) + B_i * cos(target_angle)`

4. **Validation**: Ensure output quaternion is normalized and displacement magnitude is reasonable

## Implementation Approach

The implementation will follow a modular design with clear separation of concerns:

### Core Components

**CMPDisplacementInterpolator**: Main interpolation class with geometric algorithms
- Quaternion SLERP for orientation interpolation
- Sinusoidal fitting for displacement interpolation  
- Fallback strategies for edge cases
- Performance monitoring and statistics collection

**State Class Extensions**: Add quaternion orientation support
- Optional `orientation_quaternion` field for geometric interpolation
- Backward compatibility with existing State functionality
- Conversion utilities between rotation representations

**Buffer Integration**: Store orientation data alongside existing features
- Extend `FeatureAtLocationBuffer` with quaternion storage
- Interpolation caching for performance optimization
- Integration with existing buffer operations

**Voting System Integration**: Use geometric interpolation in vote combination
- Modify `_combine_votes` in `GraphMatchingLM` to support geometric interpolation
- Fallback to linear interpolation when quaternion data unavailable
- Feature flag (`use_displacement_interpolation`) for gradual rollout

### Configuration Strategy

**Feature Flags**: Enable gradual deployment and testing
- `use_displacement_interpolation`: Master enable/disable flag
- `fallback_to_linear`: Safe fallback for compatibility
- Performance and debugging options for monitoring

**Integration with Existing Config**: Extend `MontyArgs` with `DisplacementInterpolationArgs`
- Maintains existing configuration patterns
- Allows per-experiment customization
- Conservative defaults for safe deployment

### Key Integration Points

**Spatial Arithmetic Utilities**: Leverage existing quaternion handling
- Reuse `rotations_to_quats`, `euler_to_quats` from `spatial_arithmetics.py`
- Utilize quaternion format conversion from `transform_utils.py`
- Build upon rotation alignment algorithms in `displacement_matching.py`

**Sensor Module Extensions**: Optional quaternion orientation output
- Extend `DetailedLoggingSM` with orientation estimation from pose vectors
- Configurable orientation computation methods
- Confidence estimation for displacement quality assessment

## Computational Complexity

**Per interpolation call:**
- Quaternion SLERP: O(1) - 4 quaternion operations + 1 trigonometric function
- Displacement fitting: O(1) - 3 × (2×2 matrix inversion + 2 trigonometric evaluations)
- Total: ~50-100 floating point operations vs ~10 for linear interpolation

**Memory overhead:** Minimal - no additional storage beyond input/output buffers

**Scalability:** Linear with number of sensor pairs requiring interpolation

## Integration Points

### State Class Modifications
The existing `State` class in `src/tbp/monty/frameworks/models/states.py` will be extended to support quaternion orientations:

```python
class State:
    def __init__(self, ...):
        # Existing fields...
        self.orientation_quaternion = None  # New field for quaternion representation

    def set_orientation_from_rotation(self, rotation_matrix):
        """Convert rotation matrix to quaternion for interpolation."""
        from scipy.spatial.transform import Rotation
        self.orientation_quaternion = Rotation.from_matrix(rotation_matrix)
```

### Buffer Integration
The `FeatureAtLocationBuffer` in `src/tbp/monty/frameworks/models/buffer.py` will be extended to store quaternion orientations:

```python
class FeatureAtLocationBuffer(BaseBuffer):
    def __init__(self):
        # Existing fields...
        self.orientations = {}  # New field for quaternion storage
        
    def _add_orientation_to_buffer(self, input_channel, orientation_quat):
        """Store quaternion orientation for interpolation."""
        if input_channel not in self.orientations:
            self.orientations[input_channel] = []
        self.orientations[input_channel].append(orientation_quat)
```

### Voting System Integration
The voting mechanisms in `src/tbp/monty/frameworks/models/graph_matching.py` will be updated to use the interpolator:

```python
def _combine_votes(self, votes_per_lm):
    # Existing vote combination logic...
    if self.use_displacement_interpolation:
        interpolator = CMPDisplacementInterpolator()
        # Apply geometric interpolation to displacement votes
        interpolated_displacement, interpolated_orientation = interpolator.interpolate(
            pose_1=votes_per_lm[j]["pose_with_angle"],
            pose_2=votes_per_lm[k]["pose_with_angle"], 
            target_angle=target_angle
        )
```

# Drawbacks

## Implementation Complexity
- Requires quaternion mathematics knowledge for maintenance
- Adds computational overhead compared to linear interpolation (~2-3x slower due to matrix operations)
- Uses scipy.spatial.transform (already present in codebase via motor_policies.py, spatial_arithmetics.py, and displacement_matching.py)
- Increases buffer memory usage by ~20% for quaternion storage

## Edge Cases
- **Antipodal quaternions**: When orientations are 180° apart, SLERP has ambiguous paths
- **Non-sinusoidal displacements**: Real sensor movements may not perfectly fit sine curves
- **Numerical stability**: Matrix inversion for sine fitting may be unstable in degenerate cases

## Migration Cost
- Existing experiments may show different results due to improved accuracy
- Sensor modules need updates to output quaternion orientations
- Potential breaking changes to CMP message format

# Rationale and alternatives

## Why This Design?

1. **Biological plausibility**: Avoids black-box ML approaches, maintaining interpretability
2. **Mathematical rigor**: Based on established geometric principles (SLERP, sinusoidal fitting)
3. **Minimal dependencies**: Uses only fundamental mathematical operations
4. **Clean separation**: Handles orientation and displacement independently with appropriate methods

## Alternative Approaches Considered

### Linear Vector Interpolation
- **Pros**: Simple, fast, already implemented
- **Cons**: Introduces geometric distortion, invalid for rotations
- **Verdict**: Current approach causing the problems we're solving

### ART/SIRT Iterative Solvers
- **Pros**: More accurate for complex displacement patterns
- **Cons**: Computationally expensive, overkill for typical sensor movements
- **Verdict**: Too heavy for real-time CMP communication

### Machine Learning-Based Warping
- **Pros**: Could handle arbitrary displacement patterns
- **Cons**: Black box, not biologically interpretable, requires training data
- **Verdict**: Conflicts with Monty's brain-inspired philosophy

## Impact of Not Doing This

- Continued geometric artifacts in multi-sensor scenarios
- Difficulty implementing displacement-primary CMP communication
- Suboptimal voting accuracy between learning modules
- Barriers to future work on flow detection and moving objects

# Prior art and references

## Quaternion SLERP
- Standard technique in 3D graphics and robotics
- Well-established mathematical properties and implementations
- Available in scipy.spatial.transform.Rotation.slerp()

## Sinusoidal Interpolation
- Used in tomographic reconstruction (sinogram warping)
- Classical approach in signal processing for periodic data
- Geometric interpretation aligns with sensor movement patterns

## Cortical Column Theory
- Displacement-based processing aligns with Thousand Brains Theory
- Supports the biological motivation for avoiding location-centric approaches

## Dependencies
- **scipy.spatial.transform**: Already present in tbp.monty codebase (confirmed usage in motor_policies.py, spatial_arithmetics.py, displacement_matching.py)
- **numpy**: Core dependency, already required
- **quaternion**: Already used in motor_policies.py and transform_utils.py for quaternion operations
- No new external dependencies required for this implementation

## Relationship to Existing Code
The interpolation module will leverage existing spatial arithmetic utilities:
- **spatial_arithmetics.py**: Reuse quaternion conversion functions (`rotations_to_quats`, `euler_to_quats`)
- **transform_utils.py**: Utilize quaternion format conversion utilities (`numpy_to_scipy_quat`, `scipy_to_numpy_quat`)
- **displacement_matching.py**: Build upon existing rotation alignment algorithms (`Rotation.align_vectors`)

This ensures consistency with established patterns and reduces code duplication.

# Validation Plan

## Test Scenarios

### 1. Geometric Accuracy Tests
- **Circular motion**: Two sensors at 45° and 135° on a circle, interpolate to 90°
  - Expected displacement: `[0.0, 1.0, 0.0]` (unit vector pointing up)
  - Expected orientation: 90° rotation quaternion `[0.0, 0.0, 0.707, 0.707]`
- **Helical motion**: Sensors following helical path with known analytical solution
- **Planar motion**: Sensors moving in XY plane to test Z-axis stability

### 2. Multi-Sensor Voting Tests
- **Coffee mug scenario**: Rim and handle sensors voting on object identity
  - Baseline: Current linear interpolation accuracy
  - Target: >20% improvement in spatial vote accuracy
- **Multi-finger touch**: 3+ sensors on curved surfaces
- **Cross-modal validation**: Visual and tactile sensor agreement

### 3. Regression Testing
- **Existing benchmark experiments**: YCB object recognition, texture classification
- **Performance benchmarks**: Ensure no >5% degradation in recognition accuracy
- **Memory usage**: Monitor buffer size increases from quaternion storage

### 4. Edge Case Testing
- **Antipodal quaternions**: Sensors at 180° orientations
- **Singular displacement matrices**: Sensors with identical angles
- **Numerical stability**: Extended interpolation sequences

## Success Metrics

### Accuracy Metrics
- **Geometric error**: <5% error in interpolated displacement magnitude and direction compared to analytical solutions
- **Orientation accuracy**: Quaternion interpolation error <0.1 radians from expected orientation
- **Spatial voting improvement**: >15% increase in multi-sensor voting accuracy over linear interpolation

### Performance Metrics
- **Interpolation latency**: <10ms per interpolation call on typical development hardware
- **Memory overhead**: <20% increase in buffer memory usage
- **Regression tolerance**: <5% degradation in existing benchmark accuracy

### Robustness Metrics
- **Fallback frequency**: <1% of interpolations require fallback to linear methods
- **Numerical stability**: Quaternion drift <1e-6 over 1000 interpolations
- **Edge case handling**: 100% graceful degradation for invalid inputs

# Implementation decisions

## Quaternion Normalization Strategy
**Decision**: Renormalize quaternions after each SLERP operation and log warnings if drift exceeds 1e-6. This ensures numerical stability while providing debugging information for edge cases.

## Fallback Strategies
**Decision**: Implement graceful degradation with detailed logging:
- **Singular matrix in displacement fitting**: Fall back to linear interpolation for affected axes
- **Antipodal quaternions**: Use linear interpolation with normalization when `|dot(q1, q2)| < 0.01`
- **Invalid inputs**: Return identity interpolation (first pose) with error logging

## Performance Requirements
**Decision**: Target <10ms overhead per interpolation call, measured on typical development hardware. This allows for real-time CMP communication without significant latency impact.

## Multi-Sensor Interpolation
**Decision**: For >2 sensors, use pairwise interpolation with weighted averaging:
1. Interpolate between each sensor pair
2. Weight results by sensor confidence and geometric consistency
3. Combine using quaternion averaging for orientations and vector averaging for displacements

This approach maintains the geometric principles while scaling to multiple sensors.

# Migration plan

## Phase 1: Core Implementation (Weeks 1-2)
1. **Implement CMPDisplacementInterpolator class** in new module `src/tbp/monty/frameworks/utils/displacement_interpolation.py`
2. **Add quaternion support to State class** with backward compatibility
3. **Extend FeatureAtLocationBuffer** to store orientations alongside existing data
4. **Create comprehensive unit tests** for interpolation algorithms

## Phase 2: Integration (Weeks 3-4)
1. **Update sensor modules** to output quaternion orientations where applicable
2. **Modify voting system** in graph_matching.py to use geometric interpolation
3. **Add configuration flags** to enable/disable new interpolation for testing
4. **Implement fallback mechanisms** for edge cases

## Phase 3: Validation & Optimization (Weeks 5-6)
1. **Run regression tests** on existing benchmark experiments
2. **Measure performance impact** and optimize critical paths
3. **Validate geometric accuracy** against analytical solutions
4. **Document migration guide** for external users

## Backward Compatibility Strategy
- **Feature flags**: `use_displacement_interpolation` config option (default: False initially)
- **Graceful degradation**: Automatic fallback to linear interpolation when quaternion data unavailable
- **API preservation**: Existing CMP interfaces remain unchanged, new functionality is additive

# Future possibilities

## Displacement-Primary CMP
This interpolation module provides the foundation for transitioning CMP to use displacements as primary spatial information, with locations computed as needed.

## Flow Field Integration
The sinusoidal displacement model could be extended to support flow field detection and processing, aligning with future work on [Detect Local and Global Flow](../docs/future-work/sensor-module-improvements/detect-local-and-global-flow.md).

## Hierarchical Interpolation
The approach could be extended to handle interpolation across multiple cortical column levels in hierarchical processing.

## Adaptive Interpolation
Future versions could adaptively choose between sinusoidal and other interpolation methods based on displacement pattern analysis, potentially using confidence metrics from sensor modules.

## Cross-Modal Interpolation
Extension to handle interpolation between different sensor modalities (e.g., visual and tactile) using shared displacement representations.
