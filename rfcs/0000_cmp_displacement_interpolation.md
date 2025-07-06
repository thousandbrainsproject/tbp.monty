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

3. **Displacement interpolation**: For each axis i ∈ {x, y, z}:
   - Solve linear system: `[sin(angle_1), cos(angle_1); sin(angle_2), cos(angle_2)] * [A_i; B_i] = [disp_1_i; disp_2_i]`
   - **Singular matrix handling**: If `det(matrix) < 1e-10`, fall back to linear interpolation for that axis
   - Evaluate: `disp_result_i = A_i * sin(target_angle) + B_i * cos(target_angle)`

4. **Validation**: Ensure output quaternion is normalized and displacement magnitude is reasonable

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
```

### Buffer Integration
The `FeatureAtLocationBuffer` in `src/tbp/monty/frameworks/models/buffer.py` will store quaternion orientations alongside displacement vectors for interpolation.

### Voting System Integration
The voting mechanisms in `src/tbp/monty/frameworks/models/graph_matching.py` will use the interpolator when combining votes from multiple LMs.

# Drawbacks

## Implementation Complexity
- Requires quaternion mathematics knowledge for maintenance
- Adds computational overhead compared to linear interpolation (~2-3x slower due to matrix operations)
- Uses scipy.spatial.transform (already present in codebase via motor_policies.py and spatial_arithmetics.py)

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
- **scipy.spatial.transform**: Already present in tbp.monty codebase (used in motor_policies.py, spatial_arithmetics.py)
- **numpy**: Core dependency, already required
- **quaternion**: Already used in motor_policies.py for quaternion operations
- No new external dependencies required for this implementation

# Validation Plan

## Test Scenarios
1. **Geometric accuracy tests**: Compare interpolated poses against analytical solutions for simple geometric patterns (circles, helixes)
2. **Multi-sensor voting**: Measure voting accuracy improvement in controlled multi-LM scenarios
3. **Regression testing**: Ensure existing experiments maintain comparable performance with new interpolation

## Success Metrics
- **Geometric error reduction**: <5% error in interpolated displacement magnitude and direction
- **Orientation accuracy**: Quaternion interpolation error <0.1 radians from expected orientation
- **Performance**: Interpolation overhead <10ms per vote in typical multi-sensor scenarios
- **Compatibility**: No degradation in existing benchmark experiment accuracy

# Unresolved questions

1. **Quaternion normalization**: Proposed approach - renormalize after each SLERP operation and log warnings if drift >1e-6
2. **Fallback strategies**: Proposed approach - graceful degradation to linear interpolation with logging for debugging
3. **Performance requirements**: Target <10ms overhead per interpolation call based on typical CMP message frequencies
4. **Parameter tuning**: Should sinusoidal fitting use weighted least squares based on sensor confidence?
5. **Multi-angle interpolation**: How to extend beyond two-point interpolation for scenarios with >2 sensors?

# Future possibilities

## Displacement-Primary CMP
This interpolation module provides the foundation for transitioning CMP to use displacements as primary spatial information, with locations computed as needed.

## Flow Field Integration
The sinusoidal displacement model could be extended to support flow field detection and processing.

## Hierarchical Interpolation
The approach could be extended to handle interpolation across multiple cortical column levels in hierarchical processing.

## Adaptive Interpolation
Future versions could adaptively choose between sinusoidal and other interpolation methods based on displacement pattern analysis.
