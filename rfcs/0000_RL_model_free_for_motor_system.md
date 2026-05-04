- Start Date: 2025-01-04
- RFC PR: 

NOTE: While this RFC process is document-based, you are encouraged to also include visual media to help convey your ideas.

# Summary

Replace the `JumpToGoalState` mixin in Monty's motor system with a model-free reinforcement learning (RL) agent that learns to navigate incrementally toward goal states provided by Learning Modules. Instead of teleporting the sensor to a target pose, the RL agent selects from existing Monty actions to move step-by-step toward the goal, learning from dense reward signals based on distance reduction.

My idea is to take the best practices and bring them closer to how the brain works.
Most similar in a practical sense is a hybrid of solutions:
### Episodic Memory  
- **Situation**: 'I've been in a similar situation before. What did I do then? What happened?'  
- **Algorithm**: HNSW + kNN + Gaussian Kernel Interpolation  
- **When to use**: At the begining of learning, novelty, rare / important events  
- **Characteristics**: One-shot / few-shot learning, Fast activation by similarity, High locality, poor generalization.

### Habits / skills  
- **Situation**: 'I've done this action many times under these conditions — it usually works well.'  
- **Algorithm**: Soft Actor-Critic (SAC) — parametric policy/value  
- **When to use**: During routine, automated actions, stable conditions  
- **Characteristics**: Slow learning over many repetitions, Generalization is highly effective

### Algorithm of arbitration between systems: when to trust memory and when to trust the network  
This is not a separate policy, but a mechanism for switching between behavior modes.

# Motivation

## Current Problem

The hypothesis-testing policy in Monty's Evidence Learning Module generates goal states — target poses where the sensor should move to gather disambiguating evidence about object identity. Currently, these goals are enacted by the `JumpToGoalState` mixin, which teleports the agent instantaneously to the target pose using `SetAgentPose`.
This teleportation approach has fundamental limitations as instantaneous teleportation has no biological or robotic analog. A real agent must navigate through space incrementally with collision awareness.

## Why kernel-based Q-learning with episodic memory (HNSW + kNN + Gaussian Kernel Interpolation)

### Biological Plausibility

The chosen approach (kernel-based Q-learning with episodic memory) has parallels to hippocampal memory systems:

- **Episodic storage**: Individual experiences stored as state-value pairs (analogous to hippocampal episode encoding)
- **One-shot / few-shot learning**: It is enough for a person to do something once in order to act similarly in a similar situation
- **Pattern completion**: KNN retrieval from partial state matches (analogous to hippocampal pattern completion)
- **Kernel generalization**: Smooth interpolation across similar experiences (analogous to memory generalization during retrieval)
- **Non-parametric**: No fixed-size weight matrix; memory grows with experience (analogous to ongoing hippocampal neurogenesis)

This aligns with Monty's broader goal of biologically plausible computation.

### Theoretical publications
- Ormoneit & Sen's (2002) 'Kernel-Based Reinforcement Learning' proposed and analyzed a variant of Q-learning for large state spaces, in which the Q-function is approximated by kernel regression on the data rather than by a neural network or table. A key contribution is the conditions under which such a 'kernel Q-learning' scheme converges.
- Blundell et al. (2016) 'Model-Free Episodic Control'. The agent remembers its past successful actions and returns and, when faced with a similar state, simply reproduces the best of what has already worked.

### HNSW + kNN + Gaussian Kernel Interpolation

Hierarchical Navigable Small World (HNSW) is an approximate nearest neighbor search algorithm based on a layered graph data structure. It belongs to the family of proximity graphs, where nodes (vertices) are connected based on their proximity, typically measured by the Euclidean distance.  
HNSW is currently actively used in embedding databases for searching similar text by vectors. So, I decided to use it to store and find states.  
During the learning process, the agent stores experience in a graph and then uses the weighted past experience in a similar situation. Thehnically it looks like: store point in HNSW graph, then find the K closest ones and mix them with Gaussian kernel weights.

### Why not only Deep Learing
I'm not opposed to deep learning. I agree that it's well suited for approximation, embedding, and many other tasks.  
I'm not suggesting replacing neural networks, I'm suggesting supplementing it and improving the learning process.

# Guide-level explanation

## Architecture Overview

Evidence LM's Goal-State Generator proposes the goal-state from the hypothesis-testing policy.
**goal_pose** = [x, y, z, roll, pitch, yaw]  
   ↓  
**RLGoalApproachController** computes **State Vector** using **goal_pose** as well as sensory patch input and proprioceptive information.   
   ↓  
**At the begining for new states** it needs to learn before inference  
   ↓  
**RL Q-leraning with HNSW + kNN + Gaussian Kernel Interpolation** working with discrete actions. Action Selection with **Heuristic-Guided Exploration**. Objective: To obtain smart behavior and training data for SAC. 
HNSW graph points collection (gathering experience). Copying of **successful traces into replay buffer**.
**HNSWStateStore** stores **State Vector**, actions, Q-values.
Upon reaching a certain threshold of successful validation operations moves to next step of learning.    
   ↓  
**Behavioral cloning (BC)** — method of imitation learning in which an agent learns to imitate the behavior of an expert by directly copying his actions based on data. 
Translation of discrete actions from replay buffer into continuous ones.
**Train a SAC Actor policy using supervised learning loss** to have a continuous policy that copying the behavior of a discrete policy.  
   ↓  
**Warm-start to run RL SAC training with Critic policy** as well using the Actor policy weights from the BC.
Train with a reward (progress toward the goal, penalty for collisions) in real or sim environment.
The SAC refines the policy: it makes movements smoother and more accurate, adapting to new scenes.
**Now the policy doesn't just copy of a discrete policy, it optimizes**.  
   ↓  
In future when SAC is trained we use it as **skills to propose continuous actions**  


## Advantages of this scheme:
- Quick start with Q-learning and discrete actions – no need to wait for the SAC to learn from scratch.
- Stability – Heuristic-Guided Exploration learns in new areas.
- Precision – then SAC makes movements smooth and optimal as skills.
- Biologically plausible – like human learning: first we copy, then we hone.


**Below is explanation of the main components**:

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
Update state → normalize → KNN search
→ if near existing point: update it
→ else: insert new point with interpolated init

Get state → normalize → KNN search → kernel interpolation → Q-values


## MontyActionSpace (18D)

To properly train hand, finger movement, it is suggested to use the combined agent mode (distant + surface) from the beginning.

| Index | Action               | Description                                                                 | Mode     | Parameters |
|--------|------------------------|-------------------------------------------------------------------------|-----------|-----------|
| 0–7    | MoveTangentially       | Movement tangent to the surface in 8 directions: 0°, 45°, ..., 315° | surface   | `distance: float`, `direction: VectorXYZ` |
| 8      | MoveForward            | Moving forward (in the direction the agent is looking)              | both       | `distance: float` |
| 9      | MoveForward (neg)      | Moving backward                                                          | both       | `distance: float` |
| 10     | TurnLeft               | Rotate the agent to the left (along the Y axis, yaw)                  | distant   | `rotation_degrees: float` |
| 11     | TurnRight              | Rotate the agent to the right                                            | distant   | `rotation_degrees: float` |
| 12     | LookUp                 | Tilt the agent/sensor up (pitch)                                      | distant   | `rotation_degrees: float` |
| 13     | LookDown               | Tilt the agent/sensor down                                              | distant   | `rotation_degrees: float` |
| 14     | SetSensorRotation (+)  | Rotate the sensor clockwise around the normal (yaw)                          | both       | `rotation_quat: Quaternion` |
| 15     | SetSensorRotation (-)  | Rotate the sensor counterclockwise                                          | both       | `rotation_quat: Quaternion` |
| 16     | OrientHorizontal       | Correction of position and orientation in the horizontal plane (with compensation) | surface   | `rotation_degrees: float`, `left_distance: float`, `forward_distance: float` |
| 17     | OrientVertical         | Correction of position and orientation in the vertical plane                | surface   | `rotation_degrees: float`, `down_distance: float`, `forward_distance: float` |

How to use in RL
1. Q-learning and discrete actions  
The policy outputs an index from 0 to 17.  
Fixed directions, surface_step, free_step, rotation_step are used.  
Apply masking if on_object == 0 (disallow 0-7, 16-17).  
3. Parameterized SAC (current proposal)  
The policy outputs: action index (0-17) and a continuous parameter instaed of fixed step.  
4. Purely continuous SAC (future)  
The policy outputs a vector [Δx, Δy, Δz, Δθ, Δφ] and then interprets this as a combined motion.



## RLGoalApproachController
Core RL logic — state computation, reward, collision detection, action selection with heuristic-guided exploration.

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


### Action Selection with Heuristic-Guided Exploration

#### Problem with Standard ε-Greedy

Standard ε-greedy exploration selects random actions with probability ε. In a 18-action space, a random action has only small chance of being useful (moving toward the goal). This means the most of exploration steps are wasted, resulting in slow learning and poor initial behavior.

#### My Approach: Blending Q-Values with Heuristics

Instead of random exploration, blend learned Q-values with heuristic bias derived geometric reasoning:

```python
combined = (1 - ε) × Q_normalized + ε × heuristic_normalized
action = softmax_sample(combined, temperature=max(0.1, ε))
A small fraction (ε × 10%) of actions remain purely random to guarantee full action space coverage.
```
Transition Schedule

| Phase | Epsilon | Behavior |
|---|-----------|--------|
| Cold start | 1.0 → 0.5 | Nearly pure heuristic — reasonable from step 1 |
| Learning | 0.5 → 0.1 | Blend of Q-values and heuristic |
| inference | 0.1 → 0.05 | Nearly pure Q-values with light heuristic safety net |

###	Possible Heuristic examples that use pure geometry and action space parameters

| # | Heuristic | Description |
|---|-----------|--------|
| 1 | Move toward goal | surface - MoveTangentially, distant - MoveForward |
| 2 | Goal far → fly,  goal close → crawl | surface_crawl vs free |
| 3 | Goal through surface → detach | LookUp |
| 4 | In the air navigating by rot_error | TurnRight, TurnLeft, LookDown, LookUp | 



## RLMotorPolicy
Integration point with Monty. Extends existing motor policy, activates RL when LM sends goal, falls back to standard Monty behavior otherwise.
This means:
All existing Monty behavior is preserved: When no goal is active, the parent class handles exploration (surface crawl, curvature-informed steps, orient to surface, etc.)
No modifications to LMs: Goal states are read from LM attributes that already exist for JumpToGoalState
No modifications to CMP: Reward is computed locally from proprioceptive and sensor data
Graceful degradation: If RL fails (timeout/collision), control returns to standard Monty exploration

**Inheritance**: RLMotorPolicy → SurfacePolicyCurvatureInformed → SurfacePolicy → InformedPolicy → BasePolicy → MotorPolicy




# Reference-level explanation


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
