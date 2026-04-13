- Start Date: 2025-01-04
- RFC PR: 

NOTE: While this RFC process is document-based, you are encouraged to also include visual media to help convey your ideas.

# Summary

Replace the `JumpToGoalState` mixin in Monty's motor system with a model-free reinforcement learning (RL) agent that learns to navigate incrementally toward goal states provided by Learning Modules. Instead of teleporting the sensor to a target pose, the RL agent selects from existing Monty actions (MoveTangentially, MoveForward, LookUp, etc.) to move step-by-step toward the goal, learning from dense reward signals based on distance reduction.

The system uses Q-learning with an HNSW-based state store for continuous state space support, Gaussian kernel interpolation for generalization, and heuristic-guided exploration derived from Monty's existing motor policies for efficient initial learning.

The current architecture is designed to support future addition of model-based planning. The HNSW Q-store serves as a single integration point — both model-free updates (from real experience) and model-based updates (from simulated planning) write to the same store.

World model planning would become valuable when Monty extends to:
- Object manipulation (irreversible actions)
- Active recognition (predicting sensor observations)
- Real robot deployment (expensive physical steps)
- Multi-agent coordination

# Motivation

### Current Problem

The hypothesis-testing policy in Monty's Evidence Learning Module generates goal states — target poses where the sensor should move to gather disambiguating evidence about object identity. Currently, these goals are enacted by the `JumpToGoalState` mixin, which teleports the agent instantaneously to the target pose using `SetAgentPose`.

This teleportation approach has fundamental limitations:

1. **Physically unrealistic**: Instantaneous teleportation has no biological or robotic analog. A real agent must navigate through space incrementally.

2. **No collision awareness**: The agent clips through objects when the goal is on the other side of a surface. This produces invalid sensor readings and corrupts the recognition process.

3. **No transferable motor skills**: Each teleportation is independent — the system never learns reusable navigation strategies that could transfer across objects or environments.

4. **Incompatible with embodiment**: Any future physical instantiation of Monty (robotic arm, mobile robot) cannot teleport. The motor system needs incremental navigation capabilities.

### Why Model-Free RL

The RFC description specifically calls for a model-free approach:

> "This would be model-free (i.e. no learned explicit model), but would still make use of the current sensory observations."

Model-free RL is appropriate here because:

- **No world model needed**: The agent learns state → action mappings directly from experience, without building an explicit model of object geometry or physics.
- **Sensory grounding**: Actions are chosen based on current proprioceptive state (pose error to goal) and exteroceptive observations (surface normal, curvature, depth), grounding decisions in real sensor data.
- **Incremental improvement**: The agent improves with experience, learning efficient paths and collision avoidance over time.
- **External to LMs**: The RL system lives entirely in the motor system, requiring no modifications to Learning Modules or the Cortical Messaging Protocol.

### Biological Plausibility

The chosen approach (kernel-based Q-learning with episodic memory) has parallels to hippocampal memory systems:

- **Episodic storage**: Individual experiences stored as state-value pairs (analogous to hippocampal episode encoding)
- **Pattern completion**: KNN retrieval from partial state matches (analogous to hippocampal pattern completion)
- **Kernel generalization**: Smooth interpolation across similar experiences (analogous to memory generalization during retrieval)
- **Non-parametric**: No fixed-size weight matrix; memory grows with experience (analogous to ongoing hippocampal neurogenesis)

This aligns with Monty's broader goal of biologically plausible computation.

# Guide-level explanation

## Architecture Overview

- Evidence LM's Goal-State Generator proposes the goal-state from the hypothesis-testing policy.
**goal_pose** = [x, y, z, roll, pitch, yaw]

- If LM has **goal_pose** then **RLGoalApproachController** is activated. 
**RLGoalApproachController** computes **State Vector** using **goal_pose** as well as sensory patch input and proprioceptive information.  
 
 - **HNSWStateStore** stores **State Vector**, actions, Q-values.
 
- Given **State Vector** the **RLGoalApproachController** proposes incremental actions **MontyActionSpace** to get it into the target state **goal_pose**

**Below are explanation of the main components**:

## State Vector (13D)
**Purpose**: To represent the current state of the agent relative to the goal. The state vector is a 13-dimensional vector that includes spatial and rotational errors, surface normal, and sensor information.
All spatial quantities are in the agent's local coordinate frame.

| Index | Feature | Description |
|----------|----------|----------|
| 0-2   | position_error [x, y, z]   | direction to goal in agent's local frame   |
| 3-5   | rotation_error [roll, pitch, yaw]   | orientation error (normalized angles)   |
| 6-8   | local_normal   | surface normal in agent's local frame   |
| 9   | on_object   | whether sensor on object surface   |
| 10   | alignment   | dot(goal_direction, surface_normal)   |
| 11   | distance   | Euclidean distance to goal   |
| 12   | norm_depth   | normalized depth to nearest surface   |


## HNSWStateStore

**Purpose**: Replace traditional tabular Q-table with continuous state space support using HNSW (Hierarchical Navigable Small World) graphs for fast KNN lookup and Gaussian kernel interpolation.

**Key design decisions**:
- **Why not tabular**: 13-dimensional continuous state space makes discretization impractical (exponential bin explosion). Even coarse 10-bin-per-dimension discretization yields 10^13 states.
- **Why not neural network (DQN)**: Neural function approximation suffers from catastrophic forgetting, requires GPU, needs millions of samples for convergence, and introduces the deadly triad instability (function approximation + bootstrapping + off-policy). HNSW-based approach is stable, fast to learn, and GPU-free.  
- **Theoretical basis**: Ormoneit & Sen (2002) "Kernel-Based Reinforcement Learning" — proved convergence of kernel-based Q-learning


## MontyActionSpace
**Purpose**: Map discrete Q-learning action indices to existing Monty Action objects. No custom actions — uses only what Monty already provides.

Action layout (15 discrete actions):
| Index | Feature | Description |
|----------|----------|----------|
| 0-7   | MoveTangentially   | 8 directions (0°, 45°, ..., 315°) along surface, surface_step as a config parameter   |
| 8   | MoveForward   | Fly forward (where sensor points), free_step as a config parameter    |
| 9   | MoveForward (neg)   | Fly backward, free_step as a config parameter   |
| 10   | LookUp   | Tilt sensor upward, rotation_step as a config parameter   |
| 11   | LookDown   | Tilt sensor downward, rotation_step as a config parameter   |
| 12   | SetSensorRotation (+)   | Rotate sensor clockwise, rotation_step as a config parameter   |
| 13   | SetSensorRotation (-)   | Rotate sensor counter-clockwise, rotation_step as a config parameter   |
| 14   | No-op   | Do nothing   |


Surface actions (0-7): Use Monty's tangent plane projection, ensuring movement stays ON the surface. Critical for precise positioning near the goal.  
Free actions (8-9): Enable jumping between distant surface regions (e.g., cup bottom to handle) without crawling the entire path. Without these, the agent would need 50+ surface steps for what could be a 5-step flight.  
Orient actions (10-13): Allow the agent to point the sensor toward the goal before flying, or align with the surface normal for accurate sensing.  
No-op (14): Completes the action space; rarely useful but prevents forced movement. <br>   
**Why not continuous actions**: Discrete actions with Q-learning are simpler to implement and debug. The 15-action space provides sufficient coverage. Continuous actions (via Actor-Critic) could be a future enhancement but add significant complexity.


## RLGoalApproachController
**Purpose**: Core RL logic — state computation, reward, collision detection, action selection with heuristic-guided exploration.

### Reward Function
Dense reward signal computed locally in the motor system (no LM or CMP involvement):
| Component | Reward | Done? | When |
|:----------|-------:|:-----:|:-----|
| Progress (per good step) | ~+3.0 | No | Every step; `(prev_dist - dist) / surface_step × 3.0` |
| Goal reached | +50.0 | Yes | `distance < goal_threshold (2mm)` |
| Step penalty | -0.2 | No | Every step |
| Surface violation | -5.0 | Yes | Agent passed through object (depth < 0.5mm or normal flipped) |
| Lost object (smart detach) | +0.5 | No | Lost surface but approaching goal with alignment < -0.3 |
| Lost object (drifted away) | -3.0 | No | Lost surface and moved away from goal |
| Near goal on surface | +0.5 | No | `distance < 3 × surface_step` AND `on_object = true` |
| Oscillation | -0.5 | No | Current action is opposite of previous action |
| Timeout | -10.0 | Yes | `steps >= max_steps_per_goal` |


Sensory collisions are detected from depth camera data, not Habitat physics. This matches how Monty currently handles surface interaction — through rendering, not rigid body simulation.

### Action Selection with Heuristic-Guided Exploration

#### Problem with Standard ε-Greedy

Standard ε-greedy exploration selects random actions with probability ε. In a 15-action space, a random action has only small chance of being useful (moving toward the goal). This means the most of exploration steps are wasted, resulting in slow learning and poor initial behavior.

#### Our Approach: Blending Q-Values with Heuristics

Instead of random exploration, we blend learned Q-values with heuristic bias derived from Monty's existing motor policy knowledge:

```python
combined = (1 - ε) × Q_normalized + ε × heuristic_normalized
action = softmax_sample(combined, temperature=max(0.1, ε))
A small fraction (ε × 10%) of actions remain purely random to guarantee full action space coverage.
```

Transition Schedule

| Phase | Episodes | Epsilon | Behavior |
|------------|----------|-----------|----------------------------------------------|
| Cold start | 0–50 | 1.0 → 0.5 | Nearly pure heuristic — reasonable from step 1 |
| Learning | 50–500 | 0.5 → 0.1 | Blend of Q-values and heuristic |
| Converged | 500+ | 0.1 → 0.05 | Nearly pure Q-values with light heuristic safety net |

All heuristics use pure geometry and action space parameters. No magic numbers — all thresholds are derived from surface_step and free_step.

###	Heuristic How It Works

| # | Heuristic | Source |
|---|-----------|--------|
| 1 | Move toward goal | tangent_movement() |
| 2 | Far → fly,  close → crawl | surface_crawl vs free |
| 3 | Goal through surface → detach | Geometric reasoning |
| 4 | Orient sensor to surface | orient_to_surface() | 

Example: Sensor on Cup Bottom, Goal at Handle.  
Result: agent prefers LookUp (detach from surface) → MoveForward (fly toward goal) — the geometrically optimal strategy for reaching the far side of an object.

Theoretical Basis:
This approach is supported by Hester et al. (2018) "Deep Q-learning from Demonstrations," which showed that even imperfect demonstrations dramatically accelerate RL exploration.  <br> Our heuristics serve as implicit demonstrations of reasonable motor behavior — they encode the same geometric reasoning that Monty's existing SurfacePolicyCurvatureInformed uses, but expressed as soft biases rather than hard rules.


### Training Strategy

### Training vs Inference Modes

The controller operates in three modes:

**Training mode** (`mode="train"`): Active exploration with
decaying epsilon (1.0 → 0.05). Full Q-value updates at each step.
Used when Q-store is empty or explicitly requested.

**Eval mode** (`mode="eval"`): Fixed low epsilon (0.02). Q-values
dominate action selection. Soft Q-value updates (10× slower learning
rate) allow gradual adaptation to new objects without corrupting
learned knowledge.

**Auto mode** (`mode="auto"`, default): Automatically selects
training when Q-store has fewer than 100 points, eval otherwise.
This means:
- First experiment run: trains from scratch with heuristic guidance
- Subsequent runs: immediately uses learned Q-values
- New objects: confidence mechanism naturally falls back to
  heuristic for unfamiliar states while preserving learned skills

The eval epsilon (0.02) provides a safety net: 98% learned behavior,
2% heuristic fallback. Combined with the HNSW confidence mechanism
(Q-values attenuated for unfamiliar states), this creates a
two-level protection against poor decisions in novel situations.



## RLMotorPolicy
**Purpose**: Integration point with Monty. Extends existing motor policy, activates RL when LM sends goal, falls back to standard Monty behavior otherwise.
This means:
All existing Monty behavior is preserved: When no goal is active, the parent class handles exploration (surface crawl, curvature-informed steps, orient to surface, etc.)
No modifications to LMs: Goal states are read from LM attributes that already exist for JumpToGoalState
No modifications to CMP: Reward is computed locally from proprioceptive and sensor data
Graceful degradation: If RL fails (timeout/collision), control returns to standard Monty exploration

Lifecycle hooks:
pre_episode(): Resets RL active state, preserves learned Q-values
post_episode(): Auto-saves Q-store if save directory configured, logs statistics

**Inheritance**: RLMotorPolicy → SurfacePolicyCurvatureInformed → SurfacePolicy → InformedPolicy → BasePolicy → MotorPolicy




# Reference-level explanation

## Component 1: HNSW Q-store
HNSW-based State Store for Q-Learning.
Replaces traditional hash-table Q-table with continuous state space using Hierarchical Navigable Small World graphs for fast KNN lookup and Gaussian kernel interpolation for Q-value estimation.

```python
class HNSWStateStore:
    """Q-value store using HNSW index for nearest neighbor lookup.

    Instead of discretizing the state space into bins (tabular Q-learning)
    or using a neural network (DQN), this store keeps visited states as
    points in continuous space and estimates Q-values for new states via
    Gaussian kernel interpolation over nearest neighbors.

    Key properties:
        - Linear memory growth (only visited states stored)
        - Smooth Q-function (kernel interpolation, not step function)
        - Local generalization (similar states share Q-values)
        - No catastrophic forgetting (non-parametric)
        - No GPU required

    State flow:
        get_q_values(state):
            state → normalize → KNN search → kernel interpolation → Q-values

        update_q_value(state, action, td_target):
            state → normalize → KNN search
            → if near existing point: update it
            → else: insert new point with interpolated init

    Args:
        state_dim: Dimensionality of state vector (13 for our setup).
        num_actions: Number of discrete actions (15 for our setup).
        max_points: Maximum points before eviction triggers.
        k_neighbors: Number of neighbors for KNN interpolation.
        sigma: Gaussian kernel bandwidth. Controls smoothness.
            Too small = no generalization (like tabular).
            Too large = over-smoothing (loses detail).
        insert_threshold: L2 distance below which we update existing
            point instead of inserting new one.
        evict_fraction: Fraction of points to remove when full.
        adaptive_sigma: If True, sigma adapts to local point density.
    """
```

## Component 2: Action mapping  
Monty Action Space for RL Goal Approach Controller.

Maps discrete action indices from Q-learning to Monty Action objects.
Uses existing Monty actions (MoveTangentially, MoveForward, LookUp, etc.)
instead of custom raw actions, ensuring compatibility with Monty's motor system and Habitat interface.

```python
class MontyActionSpace:
    """Maps discrete action indices to Monty Action objects.

    Action layout (15 actions total):

        SURFACE (8): Crawl along object surface
            0: MoveTangentially 0°   (forward on surface)
            1: MoveTangentially 45°  (forward-right)
            2: MoveTangentially 90°  (right)
            3: MoveTangentially 135° (backward-right)
            4: MoveTangentially 180° (backward on surface)
            5: MoveTangentially 225° (backward-left)
            6: MoveTangentially 270° (left)
            7: MoveTangentially 315° (forward-left)

        FREE (2): Fly through space
            8:  MoveForward  (forward, where sensor points)
            9:  MoveForward backward

        ORIENT (4): Change sensor orientation
            10: LookUp
            11: LookDown
            12: SetSensorRotation + (yaw clockwise)
            13: SetSensorRotation - (yaw counter-clockwise)

        META (1):
            14: no-op (do nothing)

    Why these specific actions:
        - Surface actions use Monty's tangent plane projection,
          so movement stays ON the surface automatically.
        - Free actions allow jumping between distant surface regions
          (e.g. from cup bottom to handle) without crawling all the way.
        - Orient actions let the agent point the sensor toward the
          goal before flying there.
        - No-op allows the agent to "wait" (rarely useful but
          completes the action space).

    Args:
        agent_id: Monty agent identifier for Action objects.
        surface_step: Step size for MoveTangentially (mm).
        free_step: Step size for MoveForward (mm).
        rotation_step: Step size for LookUp/Down/Rotation (degrees).
    """
```

## Component 3: RL controller
RL Goal Approach Controller.
Replaces JumpToGoalState mixin in Monty. Instead of teleporting the agent to the goal pose, this controller uses Q-learning with HNSW state store to learn incremental actions that move the agent
toward the goal.

Receives goal_pose from Evidence LM's Goal-State Generator, uses current proprioceptive state + sensor data to choose actions, and learns from dense reward (distance reduction to goal).

Handles sensory "collisions" (passing through object, losing contact) from the start, using depth camera data rather than Habitat physics.
```python
class RLGoalApproachController:
    """Q-learning controller that moves agent toward goal pose.

    Main loop (called each step by motor policy):
        1. Compute state from current pose, goal pose, sensor data
        2. Detect sensory collisions
        3. Compute reward and update Q-values (if not first step)
        4. Choose action via heuristic-guided exploration
        5. Return Monty Action object

    State vector (13D):
        local_pos_error  [3D]: direction to goal in agent's local frame
        rot_error        [3D]: orientation error (normalized angles)
        local_normal     [3D]: surface normal in agent's local frame
        on_object        [1D]: whether sensor sees object surface
        alignment        [1D]: dot(goal_direction, surface_normal)
        distance         [1D]: Euclidean distance to goal
        norm_depth       [1D]: normalized depth to nearest surface

    Args:
        agent_id: Monty agent identifier.
        config: Dictionary with optional overrides for all parameters.
    """
```

## Component 4: Monty integration
RL Motor Policy for Monty.
Extends Monty's existing SurfacePolicyCurvatureInformed to add RL-based goal approach. When Evidence LM sends a goal state, this policy uses RLGoalApproachController instead of teleporting (JumpToGoalState mixin).

When no goal is active, falls back to standard Monty motor behavior (surface crawl, curvature-informed exploration, etc.).

Integration point with Monty — this is the only module that imports from Monty's motor policy hierarchy.

```python
class RLMotorPolicy(SurfacePolicyCurvatureInformed):
    """Motor policy that uses RL for goal approach, Monty for exploration.

    Behavior modes:
        1. NO GOAL ACTIVE:
           Falls back to parent class (SurfacePolicyCurvatureInformed).
           Standard Monty exploration: surface crawl, curvature-informed
           step sizes, orient to surface, etc.

        2. GOAL ACTIVE (from LM):
           RL controller takes over. Chooses incremental actions
           (MoveTangentially, MoveForward, LookUp, etc.) to move
           agent toward goal pose.

        3. GOAL DONE (reached / collision / timeout):
           Returns to mode 1 (standard exploration).

    This design means:
        - All existing Monty behavior is preserved
        - RL only activates when LM requests goal approach
        - No modifications needed to LMs or CMP
        - Reward is computed locally in motor system

    Replaces JumpToGoalState mixin which teleports agent to goal.

    Args:
        All args from SurfacePolicyCurvatureInformed, plus:
        rl_config: Optional dict with RLGoalApproachController config.
        rl_save_dir: Optional path to save/load learned Q-values.
    """


# ══════════════════════════════════════════════════════════
# MAIN POLICY OVERRIDE
# ══════════════════════════════════════════════════════════
    def __call__(
        self,
        ctx: RuntimeContext,
        observations: Observations,
        state: MotorSystemState,
        percept: Message,
    ) -> MotorPolicyResult:
        """Main policy method called by Monty each timestep.

        Override of SurfacePolicyCurvatureInformed __call__.

        Decision flow:
            1. Check if LM has sent a goal state
            2. If yes → activate RL controller
            3. If RL active → get action from RL
            4. If RL done → deactivate, return to Monty
            5. If no goal → standard Monty behavior

        Returns:
            Return a motor policy result containing the next actions to take.
        """

```

## Possible Configuration
```python
config = {
    "motor_policy_class": "RLMotorPolicy",  # was: SurfacePolicyCurvatureInformed
    "motor_policy_args": {
        "rl_config": {
            # Q-learning
            "gamma": 0.95,             # discount factor
            "alpha": 0.1,              # learning rate
            "epsilon_start": 1.0,      # initial exploration
            "epsilon_min": 0.05,       # minimum exploration
            "epsilon_decay": 0.999,    # per-step decay

            # HNSW store
            "max_points": 50000,
            "k_neighbors": 7,
            "sigma": 1.0,
            "insert_threshold": 0.15,
            "adaptive_sigma": True,

            # Actions
            "surface_step": 5.0,   # mm
            "free_step": 10.0,     # mm
            "rotation_step": 10.0, # degrees

            # Episode
            "goal_threshold": 2.0, # mm
            "max_steps_per_goal": 100,

            # Collision detection
            "min_valid_depth": 0.5,    # mm — below = inside object
            "max_sensor_range": 100.0, # mm — max depth reading
            "normal_flip_threshold": -0.5,  # dot product for pass-through

            # Reward weights
            "reward_progress": 3.0,
            "reward_goal_reached": 50.0,
            "reward_step_penalty": -0.2,
            "reward_surface_violation": -5.0,
            "reward_smart_detach": 0.5,
            "reward_drifted away": -3.0,
            "reward_near_goal_on_surface": 0.5,
            "reward_oscillation": -0.5,
            "reward_timeout": -10.0,
        },
    },
    # ... rest of Monty config unchanged
}
```


# Drawbacks

> Why should we *not* do this? Please consider:
>
> - Implementation cost, both in terms of code size and complexity
> - Whether the proposed feature can be implemented outside of Monty
> - The impact on teaching people Monty
> - Integration of this feature with other existing and planned features
> - The cost of migrating existing Monty users (is it a breaking change?)
>
> There are tradeoffs to choosing any path. Please attempt to identify them here.

# Rationale and alternatives

## Alternatives Considered

### Alternative 1: DQN (Deep Q-Network)
***Description***: Use a neural network to approximate Q(s, a) instead of HNSW + kernel interpolation.  
***Pros***:  
- Better generalization in very high dimensions
- Mature ecosystem (PyTorch, stable-baselines3)  
***Rejected because***:  
- Requires GPU for reasonable training speed
- Catastrophic forgetting when learning new objects
- Deadly triad instability (function approximation + bootstrapping + off-policy)
- Needs millions of steps for convergence
- Not biologically plausible (backpropagation through deep network)

### Alternative 2: Actor-Critic with Continuous Actions
***Description***: Use PPO/SAC to output continuous action vectors [dx, dy, dz, droll, dpitch, dyaw].  
***Pros***:  
- No discretization artifacts (zigzag paths)
- Can move in exact direction of goal  
***Rejected because***:  
- Significantly more complex (two networks, policy gradient estimation)
- Requires GPU for reasonable training speed
- Harder to debug (continuous action space)
- Catastrophic forgetting when learning new objects
- Needs millions of steps for convergence
- Not biologically plausible (backpropagation through deep network)
- Can be added as future enhancement once discrete version is validated

### Alternative 3: Tabular Q-Learning with State Discretization
***Description***: Discretize the 13D state into bins and use standard tabular Q-learning.  
***Pros***:  
- Simplest possible implementation
- Guaranteed convergence  
***Rejected because***:  
- 13D with even 10 bins per dimension = 10^13 states (impossible)
- Coarse binning loses critical information (small position differences matter)
- No generalization between adjacent bins
- Boundary effects (state on bin edge jumps randomly between bins)


# Prior art and references

> Discuss prior art, both the good and the bad, in relation to this proposal.
> A few examples of what this can include are:
>
> - References
> - Does this functionality exist in other frameworks, and what experience has their community had?
> - Papers: Are there any published papers or great posts that discuss this? If you have some relevant papers to refer to, this can serve as a more detailed theoretical background.
> - Is this done by some other community and what were their experiences with it?
> - What lessons can we learn from what other communities have done here?
>
> This section is intended to encourage you as an author to think about the lessons from other frameworks and provide readers of your RFC with a fuller picture.
> If there is no prior art, that is fine. Your ideas are interesting to us, whether they are brand new or adaptations from other places.
>
> Note that while precedent set by other frameworks is some motivation, it does not on its own motivate an RFC.
> Please consider that Monty sometimes intentionally diverges from common approaches.

# Unresolved questions

## Open Questions
### Q1: Optimal State Dimensionality
Should we start with 13D or reduce by removing less informative features?
Is 13D enough to distinguish one state from another to train RL controller?

Several factors mitigate this:

1. **Effective dimensionality is lower**: Only 6-8 features strongly
   determine action choice. The remaining features provide refinement.

2. **States lie on trajectories**: Not randomly distributed in 13D,
   but along low-dimensional manifolds (movement paths).

3. **Feature weighting**: Critical features (distance, alignment,
   pos_error) can be upweighted in distance computation, effectively
   reducing dimensionality.

4. **Use Embeddings**: It needs a separate embedding module but we can increase dimensions without constaints risk   

### Q2: Hyperparameter and Config Sensitivity
How sensitive is the system to sigma, k_neighbors, and learning rate?
What are optimal parameters for reward weights, action steps, etc?

Current position: To be determined empirically during many tests.
Mitigation: It makes sense to develop Lightweight Enviroment with trimesh for mesh-based primitives simulation to standalone RL training without Habitat.


# Future possibilities

The current architecture is designed to support future addition of **model-based planning**. The HNSW Q-store serves as a single integration point — both model-free updates (from real experience) and model-based updates (from simulated planning) write to the same store.

World model planning would become valuable when Monty extends to:
- Object manipulation (irreversible actions)
- Active recognition (predicting sensor observations)
- Real robot deployment (expensive physical steps)
- Multi-agent coordination
