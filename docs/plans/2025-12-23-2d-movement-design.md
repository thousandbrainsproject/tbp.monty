# 2D Sensor Module Enhancements - Design Document

## Overview

Three enhancements to `TwoDPoseSM` for improved 2D surface learning:

1. **2D Movement Extraction** - Report displacement along the surface tangent plane rather than in 3D space, enabling the LM to build a 2D model of the object surface.

2. **Depth Filter for Geometric Edges** - Filter out geometric edges (object boundaries, surface creases) to retain only texture edges on the surface.

3. **Cumulative 2D Position for Graph Building** - Track and send cumulative 2D surface position to the LM, enabling true 2D graph construction where curved surfaces appear flattened (e.g., hollow cylinder → flat rectangle).

---

## Part 1: 2D Movement Extraction

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Location format | Keep full 3D location | Preserves spatial context for debugging; LM uses displacement for graph building |
| Displacement calculation | Sensor Module calculates | Encapsulates 2D behavior; LM stays agnostic to sensor type |
| Surface normal | Use current step's normal | Simpler; difference is negligible for small steps |
| Off-object behavior | Return zero displacement | Consistent with first-step; avoids None-handling |
| Displacement format | 3D vector (projected) | Compatible with existing State format; no LM changes |

### Architecture

#### Component Changes

```
TwoDPoseSM (two_d_sensor_module.py)
├── New state: _previous_location (np.ndarray or None)
├── Modified: pre_episode() - reset _previous_location to None
├── Modified: step() - calculate and set 2D displacement before returning State
└── New method: _compute_2d_displacement(current_loc, surface_normal) -> np.ndarray
```

#### Data Flow

```
step(data)
  │
  ├─► Process observations (existing)
  ├─► Extract 2D edge pose (existing)
  │
  ├─► NEW: Calculate 2D displacement
  │     ├─ If off-object or no previous: displacement = [0,0,0]
  │     ├─ Else: 3D_disp = current_loc - previous_loc
  │     └─ Project 3D_disp onto tangent plane using surface normal
  │
  ├─► state.set_displacement(projected_displacement)
  ├─► Update _previous_location = current_loc
  └─► Return State
```

### Implementation Details

#### New Instance Variable

```python
self._previous_location: Optional[np.ndarray] = None
```

#### Modified `pre_episode()`

Add reset:
```python
self._previous_location = None
```

#### New Helper Method

```python
def _compute_2d_displacement(
    self,
    current_location: np.ndarray,
    surface_normal: Optional[np.ndarray]
) -> np.ndarray:
    """Project 3D displacement onto tangent plane for 2D surface movement."""
    if self._previous_location is None or surface_normal is None:
        return np.zeros(3)

    displacement_3d = current_location - self._previous_location
    displacement_2d = project_onto_tangent_plane(displacement_3d, surface_normal)
    return displacement_2d
```

#### Modified `step()` Method

Add before returning state:
```python
# After state is constructed, before filtering/returning:
if state.get_on_object():
    surface_normal = state.get_surface_normal()
    displacement_2d = self._compute_2d_displacement(
        state.location,
        surface_normal
    )
    state.set_displacement(displacement_2d)
    self._previous_location = state.location.copy()
else:
    state.set_displacement(np.zeros(3))
    # Don't update _previous_location when off-object
```

#### Import Addition

```python
from tbp.monty.frameworks.utils.edge_detection_utils import project_onto_tangent_plane
```

### Edge Cases

| Scenario | Behavior |
|----------|----------|
| First step of episode | `_previous_location` is None → return `[0,0,0]` |
| Off-object step | Return `[0,0,0]`, don't update `_previous_location` |
| Return to object after off-object | Resume from last on-object location |
| Surface normal is invalid/None | Fallback to `[0,0,0]` |
| Large jump between steps | Works mathematically, but accuracy degrades (known caveat) |

### Testing

#### Unit Tests for `_compute_2d_displacement()`

1. **No previous location** → returns `[0,0,0]`
2. **Flat surface (normal = [0,0,1])** → displacement in x-y plane unchanged
3. **Vertical surface (normal = [1,0,0])** → x-component removed from displacement
4. **Angled surface** → verify projection math with known values
5. **None surface normal** → returns `[0,0,0]`

### Files Changed

| File | Change |
|------|--------|
| `src/tbp/monty/frameworks/models/two_d_sensor_module.py` | Add `_previous_location`, modify `pre_episode()`, modify `step()`, add `_compute_2d_displacement()` |

### Out of Scope

- Learning Module changes (none required)
- State class changes (none required)
- Config file changes (none required)
- Integration/visual tests (unit tests only)

---

## Part 2: Depth Filter for Geometric Edges

### Problem

The current edge detection (`compute_edge_features_center_weighted`) detects ALL edges in the RGB image, including:
- **Texture edges** (wanted) - e.g., letters printed on a surface
- **Geometric edges** (unwanted) - e.g., object boundary against background, surface creases

For 2D surface learning, we only want texture edges that lie ON the surface.

### Solution

Filter out edges where depth changes significantly across the edge. Texture edges have continuous depth; geometric edges have depth discontinuities.

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Filter location | Separate function after RGB edge detection | Keep RGB detection pure; easier to tune/disable |
| Depth gradient method | Sobel on depth image | Consistent with RGB approach |
| Gradient direction | Perpendicular to detected edge | Measures depth change across the edge |

### Implementation

**New function in `edge_detection_utils.py`:**

```python
def is_geometric_edge(
    depth_patch: np.ndarray,
    edge_theta: float,
    depth_threshold: float = 0.01,
) -> bool:
    """Check if detected edge is geometric (depth discontinuity) vs texture.

    Args:
        depth_patch: Depth image patch (same size as RGB patch used for edge detection)
        edge_theta: Edge tangent angle from RGB edge detection
        depth_threshold: Maximum allowed depth gradient for texture edges

    Returns:
        True if edge is geometric (should be filtered out), False if texture edge
    """
    # Compute depth gradients
    depth_dx = cv2.Sobel(depth_patch, cv2.CV_32F, 1, 0, ksize=3)
    depth_dy = cv2.Sobel(depth_patch, cv2.CV_32F, 0, 1, ksize=3)

    # Direction perpendicular to edge (normal to edge line)
    edge_normal_angle = edge_theta + np.pi / 2
    nx = np.cos(edge_normal_angle)
    ny = np.sin(edge_normal_angle)

    # Depth gradient in direction perpendicular to edge
    # Sample at patch center
    cy, cx = depth_patch.shape[0] // 2, depth_patch.shape[1] // 2
    depth_gradient_perp = abs(nx * depth_dx[cy, cx] + ny * depth_dy[cy, cx])

    return depth_gradient_perp > depth_threshold
```

**Modified `extract_2d_edge()` in `TwoDPoseSM`:**

```python
# After computing edge features from RGB:
edge_strength, coherence, theta = compute_edge_features_center_weighted(rgb_patch, ...)

if edge_strength > 0:
    # Filter out geometric edges using depth
    if is_geometric_edge(depth_patch, theta, self.depth_edge_threshold):
        # Reject this edge - it's a geometric boundary, not texture
        edge_strength = 0.0
        coherence = 0.0
        theta = 0.0
```

### Configuration

New parameter in `TwoDPoseSM.__init__()`:

```python
depth_edge_threshold: float = 0.01  # Tune based on depth sensor units/noise
```

### Edge Cases

| Scenario | Behavior |
|----------|----------|
| Depth data unavailable | Skip filter, use RGB edge as-is |
| Noisy depth sensor | May need bilateral filter on depth before gradient |
| Texture edge at geometric crease | May incorrectly filter; acceptable trade-off |

### Testing

1. **Texture edge on flat surface** → `is_geometric_edge` returns False
2. **Object boundary** → `is_geometric_edge` returns True
3. **Surface crease** → `is_geometric_edge` returns True
4. **Edge at depth discontinuity** → `is_geometric_edge` returns True

### Files Changed

| File | Change |
|------|--------|
| `src/tbp/monty/frameworks/utils/edge_detection_utils.py` | Add `is_geometric_edge()` |
| `src/tbp/monty/frameworks/models/two_d_sensor_module.py` | Add depth filter call in `extract_2d_edge()`, add config param |

---

## Part 3: Cumulative 2D Position for Graph Building

### Problem

The Part 1 implementation sets 2D displacement on the State via `state.set_displacement()`. However, the Learning Module's `_add_displacements()` method **overwrites** this displacement by recomputing it from 3D locations:

```python
# graph_matching.py - GraphLM._add_displacements()
displacement = o.location - self.buffer.get_current_location(...)  # 3D displacement
o.set_displacement(displacement)  # Overwrites our 2D displacement!
```

Additionally, the graph builder computes edge attributes from 3D locations:

```python
# object_model.py - GraphObjectModel._build_adjacency_graph()
displacements.append(locations_reduced[edge_end] - locations_reduced[edge_start])  # 3D
```

This means the learned graph contains 3D spatial relationships, not the 2D surface relationships we want.

### Goal

Enable the LM to build a **2D surface model** where:
- A hollow cylinder appears as a flat rectangle (unrolled surface)
- A sphere appears as a 2D map (with expected distortion at poles)
- Curved surfaces are "flattened" based on path integration

### Solution Overview

Track cumulative 2D position in the sensor module and send it to the LM. Two approaches:

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **3A: Dual Location** | Send both 3D location and 2D position | Preserves 3D for debugging | Requires State/LM changes |
| **3B: Replace Location** | Replace 3D location with 2D position | No LM changes needed | Loses 3D spatial context |

---

### Approach 3A: Dual Location (Keep 3D for Debugging)

Send cumulative 2D position as a separate field while preserving the original 3D location.

#### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| 2D position field | `location_2d` in State | Clear separation from 3D location |
| LM behavior | Use `location_2d` for graph building if available | Backward compatible |
| 3D location | Preserved in `location` field | Debugging, visualization |

#### Architecture Changes

```
TwoDPoseSM (two_d_sensor_module.py)
├── New state: _cumulative_2d_position (np.ndarray)
├── Modified: pre_episode() - reset _cumulative_2d_position to [0, 0]
└── Modified: step() - accumulate 2D position, set state.location_2d

State (states.py)
└── New field: location_2d (Optional[np.ndarray])

GraphLM (graph_matching.py)
└── Modified: _add_displacements() - use location_2d if available
```

#### Implementation Details

**New Instance Variable in TwoDPoseSM:**

```python
self._cumulative_2d_position: np.ndarray = np.zeros(2)
```

**Modified `pre_episode()`:**

```python
self._cumulative_2d_position = np.zeros(2)
```

**Modified `step()` Method:**

```python
# After computing 2D displacement:
if observed_state.get_on_object():
    displacement_2d = self._compute_2d_displacement(
        observed_state.location,
        surface_normal,
    )
    # Accumulate 2D position (x, y components only)
    self._cumulative_2d_position += displacement_2d[:2]

    # Set 2D location for LM graph building
    observed_state.location_2d = np.array([
        self._cumulative_2d_position[0],
        self._cumulative_2d_position[1],
        0.0
    ])

    # Keep original 3D location for debugging
    # observed_state.location remains unchanged
```

**Modified State class:**

```python
# In State.__init__():
self.location_2d: np.ndarray | None = None

# New method:
def set_location_2d(self, location_2d: np.ndarray):
    self.location_2d = location_2d
```

**Modified GraphLM._add_displacements():**

```python
def _add_displacements(self, obs):
    for o in obs:
        # Use 2D location if available (from TwoDPoseSM)
        current_loc = o.location_2d if o.location_2d is not None else o.location

        if self.buffer.get_buffer_len_by_channel(o.sender_id) > 0:
            prev_state = self.buffer.get_current_state(input_channel=o.sender_id)
            prev_loc = prev_state.location_2d if prev_state.location_2d is not None else prev_state.location
            displacement = current_loc - prev_loc
        else:
            displacement = np.zeros(3)
        o.set_displacement(displacement)
    return obs
```

#### Files Changed

| File | Change |
|------|--------|
| `src/tbp/monty/frameworks/models/two_d_sensor_module.py` | Add `_cumulative_2d_position`, set `location_2d` |
| `src/tbp/monty/frameworks/models/states.py` | Add `location_2d` field |
| `src/tbp/monty/frameworks/models/graph_matching.py` | Modify `_add_displacements()` to use `location_2d` |

---

### Approach 3B: Replace Location (Simpler, No LM Changes)

Replace the 3D location entirely with cumulative 2D position. The LM receives 2D coordinates directly.

#### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Location format | `[cum_x, cum_y, 0]` | Compatible with existing 3D location format |
| LM changes | None | Existing code works with 2D positions |
| 3D location | Optionally store as feature | For debugging if needed |

#### Architecture Changes

```
TwoDPoseSM (two_d_sensor_module.py)
├── New state: _cumulative_2d_position (np.ndarray)
├── Modified: pre_episode() - reset _cumulative_2d_position to [0, 0]
└── Modified: step() - replace state.location with [cum_x, cum_y, 0]
```

#### Implementation Details

**New Instance Variable:**

```python
self._cumulative_2d_position: np.ndarray = np.zeros(2)
```

**Modified `pre_episode()`:**

```python
self._cumulative_2d_position = np.zeros(2)
```

**Modified `step()` Method:**

```python
# After computing 2D displacement:
if observed_state.get_on_object():
    displacement_2d = self._compute_2d_displacement(
        observed_state.location,
        surface_normal,
    )
    # Accumulate 2D position
    self._cumulative_2d_position += displacement_2d[:2]

    # Optionally store original 3D location as feature for debugging
    if "location_3d" in self.features:
        observed_state.non_morphological_features["location_3d"] = observed_state.location.copy()

    # Replace location with cumulative 2D position
    observed_state.location = np.array([
        self._cumulative_2d_position[0],
        self._cumulative_2d_position[1],
        0.0
    ])
```

#### Files Changed

| File | Change |
|------|--------|
| `src/tbp/monty/frameworks/models/two_d_sensor_module.py` | Add `_cumulative_2d_position`, replace `location` |

#### Visualization

With Approach 3B, the learned graph positions are already 2D. Visualization is straightforward:

```python
def plot_2d_learned_surface(graph_model):
    """Plot the learned 2D surface model."""
    pos = graph_model.pos.numpy()  # Already [x, y, 0]

    plt.figure(figsize=(8, 8))
    plt.scatter(pos[:, 0], pos[:, 1], c=range(len(pos)), cmap='viridis')
    plt.xlabel("Cumulative X")
    plt.ylabel("Cumulative Y")
    plt.title("Learned 2D Surface Model")
    plt.axis('equal')
    plt.show()
```

---

### Comparison

| Aspect | Approach 3A (Dual Location) | Approach 3B (Replace Location) |
|--------|----------------------------|-------------------------------|
| LM changes | Required | None |
| State changes | Required | None |
| 3D debugging | Built-in | Optional (as feature) |
| Complexity | Higher | Lower |
| Backward compatibility | Full | May affect 3D-dependent code |
| Recommended for | Production use | Quick verification |

### Recommendation

- Start with **Approach 3B** for initial verification that 2D path integration works correctly
- Move to **Approach 3A** for production if 3D debugging context is needed

---

## Part 4: Curvature-Based Arc Length Correction

### Problem

The tangent-plane projection in Parts 1-3 computes chord length, not arc length. On curved surfaces, this **underestimates** the true distance traveled:

```
projected_length = chord × cos(θ)  <  arc_length
```

For a cylinder with radius 5cm and a 2mm step, the underestimation is ~0.04%. Small per step, but accumulates over an episode.

**Goal**: Enable geometry-invariant pattern recognition - a logo learned on a flat surface should match when encountered on a curved mug.

### Mathematical Foundation

For a circular arc with curvature κ (1/radius) and chord length c:

```
arc_length = (2/κ) × arcsin(κc/2)
```

The relevant curvature is the **normal curvature in the direction of movement**, computed from principal curvatures k1, k2 using Euler's formula:

```
κ_direction = k1×cos²(φ) + k2×sin²(φ)
```

where φ is the angle between the movement direction and the first principal curvature direction.

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Correction formula | Exact arcsin | Handles high curvature correctly; one arcsin per step is negligible cost |
| Small curvature fallback | Skip correction when \|κc\| < 0.001 | Avoids numerical issues; already accurate |
| Curvature source | Principal curvatures from depth | Already computed by `HabitatObservationProcessor` |
| Direction-dependent | Yes, use Euler's formula | Movement along vs across curvature matters |

### Architecture

```
TwoDPoseSM (two_d_sensor_module.py)
├── Modified: __init__() - add use_arc_length_correction flag
├── Modified: step() - apply correction after projection
└── New method: _apply_arc_length_correction(projected_disp, curvature_info) -> np.ndarray

edge_detection_utils.py
└── New function: compute_arc_length_correction(chord_length, curvature) -> float
```

### Implementation Details

#### New Utility Function

```python
def compute_arc_length_correction(
    chord_length: float,
    curvature: float,
    threshold: float = 0.001,
) -> float:
    """Compute arc length from chord length using surface curvature.

    For a circular arc: arc = (2/κ) × arcsin(κc/2)

    Args:
        chord_length: Projected displacement magnitude (chord of the arc).
        curvature: Normal curvature in the direction of movement (1/radius).
        threshold: Skip correction when |κ×c| < threshold (already accurate).

    Returns:
        Estimated arc length. Returns chord_length unchanged if curvature
        is negligible or would cause numerical issues.
    """
    kc = abs(curvature * chord_length)

    if kc < threshold:
        # Curvature effect negligible, chord ≈ arc
        return chord_length

    if kc >= 2.0:
        # Chord longer than diameter - invalid geometry, skip correction
        return chord_length

    # arc = (2/κ) × arcsin(κc/2)
    arc_length = (2.0 / abs(curvature)) * np.arcsin(kc / 2.0)
    return arc_length
```

#### Modified `_compute_2d_displacement()` Signature

```python
def _compute_2d_displacement(
    self,
    current_location: np.ndarray,
    surface_normal: np.ndarray | None,
    principal_curvatures: np.ndarray | None = None,  # NEW: [k1, k2]
    curvature_directions: np.ndarray | None = None,  # NEW: [dir1, dir2]
) -> np.ndarray:
```

#### Modified `step()` Method

```python
# After computing projected displacement:
if self.use_arc_length_correction and observed_state.get_on_object():
    principal_curvatures = observed_state.non_morphological_features.get(
        "principal_curvatures"
    )
    curvature_directions = observed_state.morphological_features.get(
        "pose_vectors"
    )  # [normal, dir1, dir2]

    displacement_2d = self._compute_2d_displacement(
        observed_state.location,
        surface_normal,
        principal_curvatures=principal_curvatures,
        curvature_directions=curvature_directions[1:] if curvature_directions else None,
    )
```

#### Direction-Dependent Curvature Calculation

```python
def _get_directional_curvature(
    self,
    movement_direction: np.ndarray,
    k1: float,
    k2: float,
    dir1: np.ndarray,
    dir2: np.ndarray,
) -> float:
    """Compute normal curvature in the direction of movement using Euler's formula.

    κ_direction = k1×cos²(φ) + k2×sin²(φ)

    where φ is angle between movement direction and first principal direction.
    """
    # Project movement onto tangent plane (should already be in-plane)
    movement_normalized = normalize(movement_direction)

    # Angle between movement and first principal direction
    cos_phi = np.clip(np.dot(movement_normalized, dir1), -1.0, 1.0)
    cos_phi_sq = cos_phi ** 2
    sin_phi_sq = 1.0 - cos_phi_sq

    return k1 * cos_phi_sq + k2 * sin_phi_sq
```

### Configuration

New parameter in `TwoDPoseSM.__init__()`:

```python
use_arc_length_correction: bool = False  # Off by default for backward compatibility
```

Requires `principal_curvatures` in features list when enabled.

### Edge Cases

| Scenario | Behavior |
|----------|----------|
| Flat surface (k1 ≈ k2 ≈ 0) | Correction factor ≈ 1.0, no change |
| Curvature data unavailable | Fall back to uncorrected chord length |
| Very high curvature (κc ≥ 2) | Skip correction (invalid geometry) |
| Movement along zero-curvature direction | κ_direction = 0, no correction |
| Saddle point (k1 > 0, k2 < 0) | Uses signed curvature in movement direction |

### Testing

1. **Flat surface** → correction factor = 1.0 (no change)
2. **Cylinder, movement along axis** → correction factor = 1.0 (k_direction = 0)
3. **Cylinder, movement around circumference** → arc > chord by expected amount
4. **Sphere** → uniform correction regardless of direction
5. **Known geometry** → verify arc length matches analytical solution

### Files Changed

| File | Change |
|------|--------|
| `src/tbp/monty/frameworks/utils/edge_detection_utils.py` | Add `compute_arc_length_correction()` |
| `src/tbp/monty/frameworks/models/two_d_sensor_module.py` | Add `use_arc_length_correction` param, modify `_compute_2d_displacement()`, add `_get_directional_curvature()` |

### Dependencies

- Requires `principal_curvatures` feature enabled in `HabitatObservationProcessor`
- Requires `pose_vectors` (curvature directions) for direction-dependent correction

---

## Part 5: Local Tangent Coordinate System with Axis Stabilization

### Problem

Part 3B accumulates 2D position by summing world X/Y components of projected displacements:

```python
displacement_2d = project_onto_tangent_plane(displacement_3d, surface_normal)
self._cumulative_2d_position += displacement_2d[:2]  # World X, Y - BUG!
```

On curved surfaces like cylinders, the tangent plane **rotates** as the sensor moves. Summing world coordinates produces distorted "hourglass" shapes instead of rectangles because:

1. At the "front" of a cylinder, circumferential movement maps to world X
2. At the "side", the same physical movement maps to world Y
3. Summing these inconsistent coordinates creates geometric distortion

### Solution: Local Tangent Coordinates

Express each displacement in the **tangent plane's local coordinate system** using the tangent directions (t1, t2) from `pose_vectors`, then accumulate:

```python
# Project displacement onto tangent plane (world coords)
disp_world = project_onto_tangent_plane(displacement_3d, normal)

# Decompose into local tangent coordinates
u = np.dot(disp_world, t1)  # Component along first tangent direction
v = np.dot(disp_world, t2)  # Component along second tangent direction

# Accumulate in local coords (consistent "unrolled" frame)
cumulative_2d_position += [u, v]
```

### Challenge: Axis Instability

The tangent directions from `pose_vectors` can be unstable between steps:

| Issue | Description | Example |
|-------|-------------|---------|
| **Sign flip** | t1 can arbitrarily flip to -t1 | Edge detection finds same edge from opposite side |
| **Axis swap** | t1 and t2 can swap roles | Near umbilical points where k1 ≈ k2 |
| **Source change** | Edge-based vs curvature-based tangents | `pose_from_edge` flag changes |

Without stabilization, these discontinuities cause sudden jumps in accumulated position.

### Stabilization Algorithm

Maintain previous frame's tangent directions and align new directions to maximize continuity:

```python
def _stabilize_tangent_frame(
    self,
    t1_new: np.ndarray,
    t2_new: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Stabilize tangent frame against sign flips and axis swaps.

    Ensures continuity with previous frame by:
    1. Checking for axis swap (t1/t2 roles exchanged)
    2. Correcting sign flips (t1 vs -t1)

    Args:
        t1_new: First tangent direction from current pose_vectors.
        t2_new: Second tangent direction from current pose_vectors.

    Returns:
        Stabilized (t1, t2) that maximizes alignment with previous frame.
    """
    if self._previous_t1 is None:
        # First step - no previous frame to align to
        return t1_new, t2_new

    # Compute alignment scores
    dot_t1_t1 = np.dot(t1_new, self._previous_t1)  # t1 aligned with prev t1
    dot_t1_t2 = np.dot(t1_new, self._previous_t2)  # t1 aligned with prev t2
    dot_t2_t1 = np.dot(t2_new, self._previous_t1)  # t2 aligned with prev t1
    dot_t2_t2 = np.dot(t2_new, self._previous_t2)  # t2 aligned with prev t2

    # Check for axis swap: if t1_new aligns better with prev_t2 than prev_t1
    if abs(dot_t1_t2) > abs(dot_t1_t1):
        # Swap detected - exchange t1 and t2
        t1_new, t2_new = t2_new, t1_new
        dot_t1_t1 = dot_t2_t1
        dot_t2_t2 = dot_t1_t2

    # Correct sign flips
    if dot_t1_t1 < 0:
        t1_new = -t1_new
    if dot_t2_t2 < 0:
        t2_new = -t2_new

    return t1_new, t2_new
```

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Reference frame | First step's tangent frame | Simple, deterministic starting point |
| Stabilization trigger | Every step | Prevents drift accumulation |
| Swap detection | Compare cross-alignment magnitudes | Handles gradual rotation |
| Sign correction | Enforce positive dot product with previous | Maintains consistent orientation |
| Fallback when no pose_vectors | Use world X/Y (original behavior) | Graceful degradation |

### Architecture

```
TwoDPoseSM (two_d_sensor_module.py)
├── New state: _previous_t1, _previous_t2 (np.ndarray or None)
├── Modified: pre_episode() - reset tangent frame state
├── New method: _stabilize_tangent_frame(t1, t2) -> tuple[np.ndarray, np.ndarray]
├── New method: _to_local_tangent_coords(disp_world, t1, t2) -> np.ndarray
└── Modified: step() - use local coords for accumulation
```

### Implementation Details

#### New Instance Variables

```python
# For tangent frame stabilization (Part 5)
self._previous_t1: np.ndarray | None = None
self._previous_t2: np.ndarray | None = None
```

#### Modified `pre_episode()`

```python
def pre_episode(self):
    # ... existing resets ...
    self._previous_t1 = None
    self._previous_t2 = None
```

#### New Method: `_to_local_tangent_coords()`

```python
def _to_local_tangent_coords(
    self,
    displacement_world: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
) -> np.ndarray:
    """Express world-space displacement in local tangent coordinates.

    Args:
        displacement_world: Displacement vector in world coordinates
            (should already be in the tangent plane).
        t1: First tangent direction (unit vector).
        t2: Second tangent direction (unit vector).

    Returns:
        Array [u, v] where u is component along t1, v along t2.
    """
    u = np.dot(displacement_world, t1)
    v = np.dot(displacement_world, t2)
    return np.array([u, v])
```

#### Modified `step()` Method

```python
# Calculate 2D displacement and accumulate cumulative 2D position
if observed_state.get_on_object():
    surface_normal = observed_state.get_surface_normal()

    # Get tangent frame from pose_vectors
    pose_vectors = observed_state.morphological_features.get("pose_vectors")
    use_local_coords = pose_vectors is not None and len(pose_vectors) >= 3

    if use_local_coords:
        t1_raw, t2_raw = pose_vectors[1], pose_vectors[2]
        t1, t2 = self._stabilize_tangent_frame(t1_raw, t2_raw)

    # Compute displacement (with optional arc length correction)
    displacement_2d = self._compute_2d_displacement(
        observed_state.location,
        surface_normal,
        principal_curvatures=principal_curvatures,
        curvature_direction=t1 if use_local_coords else None,
    )

    observed_state.set_displacement(displacement_2d)
    self._previous_location = observed_state.location.copy()

    # Accumulate in local tangent coords (Part 5) or world coords (fallback)
    if use_local_coords:
        local_disp = self._to_local_tangent_coords(displacement_2d, t1, t2)
        self._cumulative_2d_position += local_disp

        # Update tangent frame for next step
        self._previous_t1 = t1.copy()
        self._previous_t2 = t2.copy()
    else:
        # Fallback to world coords (original behavior)
        self._cumulative_2d_position += displacement_2d[:2]

    # Replace 3D location with cumulative 2D position
    observed_state.location = np.array([
        self._cumulative_2d_position[0],
        self._cumulative_2d_position[1],
        0.0
    ])
```

### Edge Cases

| Scenario | Behavior |
|----------|----------|
| First step of episode | No stabilization needed; use raw tangent dirs |
| pose_vectors unavailable | Fall back to world X/Y coordinates |
| t1 and t2 nearly parallel | Stabilization still works; decomposition may be ill-conditioned |
| Rapid 180° rotation | Stabilization tracks incrementally; large single-step rotations may fail |
| pose_from_edge changes | Stabilization handles transition smoothly |

### Geometric Interpretation

After this fix, the learned surface coordinates represent:

```
          Cylinder "unrolled"
    +---------------------------+
    |                           |
    |     u (around circumf.)   |
    |  -----------------------> |
    |  |                        |
    |  | v (along axis)         |
    |  |                        |
    |  v                        |
    +---------------------------+
```

- **u-axis**: Aligns with circumferential direction (first tangent)
- **v-axis**: Aligns with axial direction (second tangent)
- Curved surface is "unrolled" into a flat rectangle

### Testing

1. **Cylinder surface** → learned model should be rectangular, not hourglass
2. **Sign flip injection** → manually flip t1 sign, verify stabilization corrects it
3. **Axis swap injection** → swap t1/t2, verify stabilization detects and fixes
4. **Flat surface** → behavior unchanged (tangent frame is constant)
5. **No pose_vectors** → graceful fallback to world coords

### Files Changed

| File | Change |
|------|--------|
| `src/tbp/monty/frameworks/models/two_d_sensor_module.py` | Add `_previous_t1`, `_previous_t2`, `_stabilize_tangent_frame()`, `_to_local_tangent_coords()`, modify `step()` |

### Dependencies

- Requires `pose_vectors` in features for local coordinate mode
- Works with both edge-based and curvature-based pose_vectors

---

## Known Caveats

From the original 2D SM outline (`docs/2d_sm.md`):

1. ~~Path integration only works locally - accuracy degrades for large movements~~ **Mitigated by Part 4** (arc length correction)
2. No representation of returning to the same place after a full circle
3. ~~Learned model may vary based on exploration path taken~~ **Mitigated by Part 5** (consistent local coordinates)
4. Cumulative 2D position drifts over time due to sensor noise

**Additional caveats:**

5. Arc length correction assumes locally circular curvature - accuracy degrades for rapidly varying curvature (e.g., sharp corners) [Part 4]
6. Tangent frame stabilization may fail for large single-step rotations (>90 degrees) [Part 5]
7. Near umbilical points (k1 ≈ k2), axis swap detection becomes ambiguous [Part 5]
