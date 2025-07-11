- Start Date: 2025-07-08
- RFC PR: (leave this empty, it will be filled in after RFC is merged)

# Intelligent Resampling in Monty

> [!NOTE]
> This PR is in Draft mode and actively being refined. Your comments and feedback are welcome at any stage! If you prefer to wait for the complete (& coherent) version, please check back when the PR status changes to Open. Thank you!

This is a high-level RFC on intelligence resampling in Monty, considering the below three questions:

1. How can we re-anchor hypotheses to model points for robustness to noise and distortions?
2. How can we implement and test resampling informed by out-of-reference-frame observations?
3. How can we use prediction errors in off-object observations to eliminate hypotheses? 

## Basic Understanding

### Current State of Hypotheses in Monty

Since we will be dealing with hypotheses a lot in this RFC, I think it is worth going over the current structure of Hypotheses and how they are used in `tbp.monty` today. 

Currently, there is a `Hypotheses` dataclass (`src/tbp/monty/frameworks/models/evidence_matching/hypotheses.py`):
```python
@dataclass
class Hypotheses:
    """Set of hypotheses consisting of evidence, locations, and poses.

    The three arrays are expected to have the same shape. Each index corresponds to a
    hypothesis.
    """

    evidence: np.ndarray # numpy array of confidence scores
    locations: np.ndarray # numpy array of 3D positions in the object's reference frame
    poses: np.ndarray # numpy array of 3x3 rotation matrices, shape
```

Hypotheses are stored in `EvidenceGraphLM` class for each object or `graph_id`. The `graph_id` serves as `keys` for the below dictionaries:
- `self.evidence[graph_id]` = Evidence scores array
- `self.possible_locations[graph_id]` = Location array
- `self.possible_poses[graph_id]` = Pose array

To update an hypothesis, we have the following `HypothesesUpdater` Protocol:
```python
class HypothesesUpdater(Protocol):
    def update_hypotheses(
        self,
        hypotheses: Hypotheses,
        features: dict,
        displacements: dict | None,
        graph_id: str,
        mapper: ChannelMapper,
        evidence_update_threshold: float,
    ) -> list[ChannelHypotheses]:
```

A specific Protocol called `DefaultHypothesesUpdater` does the following in its `update_hypotheses` method:
```python
"""Update hypotheses based on sensor displacement and sensed features.

    Updates existing hypothesis space or initializes a new hypothesis space
    if one does not exist (i.e., at the beginning of the episode). Updating the
    hypothesis space includes displacing the hypotheses possible locations, as well
    as updating their evidence scores. This process is repeated for each input
    channel in the graph.

    Args:
        hypotheses: Hypotheses for all input channels in the graph_id
        features: Input features
        displacements: Given displacements
        graph_id: Identifier of the graph being updated
        mapper: Mapper for the graph_id to extract data from
            evidence, locations, and poses based on the input channel
        evidence_update_threshold: Evidence update threshold.

    Returns:
        The list of hypotheses updates to be applied to each input channel.
"""
```
Summary:
1. If there are no hypotheses, make some. 
2. If there are hypotheses, _displace_, i.e. delete old hypotheses and fill with new hypotheses. 

Let's look at the key `displace_hypotheses_and_compute_evidence()` method from `DefaultHypothesesDisplacer` class. From the first two lines, we can already see it calculates the possible locations the hypotheses will be updated to, i.e. the variable `search_locations`. 
```python
# Have to do this for all hypotheses so we don't loose the path information
rotated_displacements = possible_hypotheses.poses.dot(channel_displacement)
search_locations = possible_hypotheses.locations + rotated_displacements
```

If we have noise in the `possible_hypotheses.poses` and `possible_hypotheses.locations`, then we will have noise in the `search_locations`. 

```math
\mathbf{R}_i' = \mathbf{R}_i + \boldsymbol{\epsilon}_{\mathbf{R}_i} \\
\mathbf{p}_i' = \mathbf{p}_i + \boldsymbol{\epsilon}_{\mathbf{p}_i} \\
\mathbf{d}_i^{\text{rot}} = \mathbf{R}_i' \mathbf{d}^{\text{channel}} \\
\mathbf{s}_i = \mathbf{p}_i' + \mathbf{d}_i^{\text{rot}}
```

Combined together: 
```math
\mathbf{s}_i = \mathbf{R}_i \mathbf{d}^{\text{channel}} + \mathbf{p}_i + \boldsymbol{\epsilon}_i \\
\mathbf{s}_i \sim \mathcal{N}(\mathbf{R}_i \mathbf{d}^{\text{channel}} + \mathbf{p}_i, \sigma^2 \mathbf{I})
```
**Note**: There is an assumption that the noise added is Gaussian, which is reasonable for positions but less so for Rotations (SO(3) space). It may still be reasonable if we assume noise added to rotation is small. 


Explaining the rest of the code:
1. If there are no hypotheses whose `possible_hypotheses.evidence => evidence_udpate_threshold`, then we don't do anything to the evidence scores and return hypotheses just with the `search_locations` updated. 
2. If there are, then we update evidence scores for these hypotheses using `_calculate_evidence_for_new_locations`. 

A look at the method `_calculate_evidence_for_new_locations()`:
```python
"""Use search locations, sensed features and graph model to calculate evidence.

    First, the search locations are used to find the nearest nodes in the graph
    model. Then we calculate the error between the stored pose features and the
    sensed ones. Additionally we look at whether the non-pose features match at the
    neighboring nodes. Everything is weighted by the nodes distance from the search
    location.
    If there are no nodes in the search radius (max_match_distance), evidence = -1.

    We do this for every incoming input channel and its features if they are stored
    in the graph and take the average over the evidence from all input channels.

    Returns:
        The location evidence.
"""
```
Again, because the method depends on `search_locations`, we need a way to account for this when there is sensor noise added. 


## How can we re-anchor hypotheses to model points for robustness to noise and distortions?

In experiments where we have sensor noise, we may wish to "re-anchor" hypotheses to account for accumulated drift and distortions. By "re-anchoring", we mean updating the `locations` and `poses` attributes of the `Hypotheses` object to align with the underlying discretized model structure.

### Back of the Envelop calculation for Sensor Noise

- **Location noise**: 2mm Gaussian ($\sigma$ = 0.002m)
- **Voxel size**: 6mm (0.3m max_size ÷ 50 voxels_per_dim)
- **Noise-to-voxel ratio**: 33% - significant relative to discretization
- **Rotation noise**: $2\degree$ per axis ($\sigma = 2.0\degree$)
- **Rotation grid**: $45\degree$ intervals
- **Noise-to-grid ratio**: 4.4% - more tolerant than location

Without re-anchoring, drift accumulates as a random walk: $ 2 \times \sqrt{N}$ where $N$ is the number of steps taken. 
- After 10 steps: ~6.3mm location drift (exceeds voxel size)
- After 100 steps: ~20mm drift (over 3 voxels)

### Location Re-anchoring Strategies


#### 1. Maximum Likelihood Estimation (MLE) Approach

For Gaussian noise $\mathcal{N}(0, \sigma^2)$, the MLE of true location from N observations is their weighted mean, with variance reduced by factor of $N$. From the equations above, i.e. $\mathbf{s}_i \sim \mathcal{N}(\mathbf{R}_i \mathbf{d}^{\text{channel}} + \mathbf{p}_i, \sigma^2 \mathbf{I})$, the optimal estimator of true location would be to "simply" find the mean of several noisy locations. The approach is similar to how we are taking weighted sum of evidence. The basic idea would be to consolidate nearby hypotheses using evidence-weighted averaging. There are several implementation details to consider, such as :

1. How to threshold for high-evidence hypotheses?
2. How to group or cluster them (e.g. clustering can be expensive, and maybe we just want to have a simple heuristic such as $\epsilon$-radius of the location of MLH)?
3. Whether to consider hypotheses from different objects?

A downstream RFC or PR should consider these questions if implementing. 

**Note**: For noise in rotation (which is happening in SO(3) space and not Euclidean space), we should use a more "general" mean such as [this](https://en.wikipedia.org/wiki/Fr%C3%A9chet_mean). 

#### 2. Voxel Center Re-anchoring Approach

This approach leverages knowledge of the model's voxel structure in `GridObjectModels`. This would basically "snap" the location to the voxel center. This idea was inspired from `how-learning-modules-work.md`:

> If max_voxels_per_dim would be set to 4 (as shown in this figure), then each voxel would be of size **2**cm<sup>3</sup> and any locations within that voxel cannot be distinguished.


```python
# some pseudocode
def voxel_center_reanchoring(self, confidence_threshold=0.8):
    """Snap high-confidence hypotheses to nearest voxel centers."""
    voxel_size = self.graph_memory.max_graph_size / self.graph_memory.num_model_voxels_per_dim
    
    for graph_id in self.get_all_known_object_ids():
        evidences = self.evidence[graph_id]
        locations = self.possible_locations[graph_id]
        
        # Only re-anchor high-confidence hypotheses
        high_conf_mask = evidences > confidence_threshold * np.max(evidences)
        
        for idx in np.where(high_conf_mask)[0]:
            location = locations[idx]
            
            # Compute nearest voxel center
            voxel_indices = np.round(location / voxel_size)
            voxel_center = voxel_indices * voxel_size
            
            # Only snap if within half voxel (avoid wrong-voxel errors)
            if np.linalg.norm(location - voxel_center) < voxel_size / 2:
                self.possible_locations[graph_id][idx] = voxel_center
```

The benefit of this approach is that it should be faster than the MLE approach above, however, I'm not sure if voxel_size would be considered priviledged information to the LM (certainly we can reach for this information in software, but doesn't mean we _should_ do that.)

Finally, neither of the above two approaches are mutually exclusive, and may be beneficial to implement a hybrid approach combining the two.

## How can we use prediction errors in off-object observations to eliminate hypotheses? 

**Thought:** When a hypothesis predicts we should be on-object but we observe off-object (or vice versa), this error provides a strong evidence against that hypothesis. Let me think about the potential prediction error types:

1. **On/Off Object Error**: A binary error where the hypothesis expected on-object but observed off-object
2. **Feature Error**: Expected certain features but received "empty" or background features
3. **Distance Error**: Expected distance vs. actual distance 

This may be a slightly far-fetched idea, but possibly restructuring the `Hypotheses` class may open avenues to address this question (and possibly better suited for modeling compositional objects). 

### Encapsulate Object_ID in Hypotheses class
Instead of the current following structure where we have a hypothesis for each `graph_id`, perhaps we can try to include this information into the hypothesis itself, e.g.:
```python
# Option #1
@dataclass
class Hypotheses:
    object_id: int # guess for what object it thinks it is, maybe like -1 if it doesn't think it's on object
    evidence: np.ndarray # numpy array of confidence scores
    locations: np.ndarray # numpy array of 3D positions in the object's reference frame
    poses: np.ndarray # numpy array of 3x3 rotation matrices
```
Or we can keep the existing structure of a hypothesis for each `graph_id`, but have some way to compute the probability that it _thinks_ it is on object, e.g.:
```python
# Option 2
@dataclass
class Hypotheses:
    on_object_probability: np.ndarray
    evidence: np.ndarray # numpy array of confidence scores
    locations: np.ndarray # numpy array of 3D positions in the object's reference frame
    poses: np.ndarray # numpy array of 3x3 rotation matrices
```
Option 3: Combination of above with five fields including both `object_id` and `on_object_probability`.  

With Option 1, the `EvidenceGraphLM` could store list of `Hypotheses` instead of primitives such as `self.evidence[graph_id]`. This is just a hunch at this point, but I can better imagine a child `EvidenceGraphLM` and parent `EvidenceGraphLM` storing two different `object_id`s but sharing the same location and poses. It's also a bit easier for me to imagine how voting may work in this manner. 

With Option 2, the `on_object_probability` becomes a metric for which we can eliminate hypotheses, e.g. if it's too small, get rid of the hypothesis. Implementation-wise, there are several ways we can do this using distance-based or feature-based (or better yet, combination thereof), e.g.

```python
# some pseudocode for distance-based probability
distance_to_surface = some_distance_function_like_L2(hypothesis.locations, nearest_node.location)
hypothesis.on_object_probability = np.exp(-distance_to_surface / maybe_some_scaling_factor) # if far away, then the probability that hypothesis is on object gets exponentially smaller
```

```python
# some pseudocode for feature-based probability
# 1. Feature presence check
has_features = not self._is_empty_features(observed_features)
feature_presence_prob = 1.0 if has_features else 0.1

# 2. Feature similarity if features exist
if has_features and expected_features is not None:
    # Compute similarity across different feature types
    similarities = {}

    if 'color' in observed_features and 'color' in expected_features:
        color_sim = self._compute_color_similarity(
            observed_features['color'],
            expected_features['color']
        )
        similarities['color'] = color_sim

    if 'texture' in observed_features and 'texture' in expected_features:
        texture_sim = self._compute_texture_similarity(
            observed_features['texture'],
            expected_features['texture']
        )
        similarities['texture'] = texture_sim

    if 'curvature' in observed_features and 'curvature' in expected_features:
        curv_sim = self._compute_curvature_similarity(
            observed_features['curvature'],
            expected_features['curvature']
        )
        similarities['curvature'] = curv_sim

    # Weighted average of similarities
    if similarities:
        feature_match_prob = np.mean(list(similarities.values()))
```

## How can we implement and test resampling informed by out-of-reference-frame observations?

(While this was the second question, I put it after the question regarding eliminating hypothesis because I think this is an extension, i.e. instead of eliminating, how can we re-anchor the hypothesis based on off-object observations. Please let me know if I'm understanding the problem correctly! :) I suppose it also goes back to the first question of how can we re-anchor or update hypothesis.)

Some related questions I'd like to think about:
  1. What can off-object information tell us? 

Off-object observations provides negative space information. This creates boundary constraints over time.

  2. When we're learning an object, we don't have a complete graph model. How do we deal with this?

Early in learning, our object model is incomplete, and we need to distinguish between: (1) being off-object (beyond true boundary), (2) being on unexplored part of object, and maybe (3) we are in some concave region or hole?

Also related to compositional objects, we may need a more principled approach to decide when to: (1) extend the existing object model, (2) decided we found a boundary, and (3) recognize we transitioned to a different object.

On a tangential thought, finding "boundaries" via off-object observations could also be great to guide policies, though I won't delve into policies in this RFC.

### More thoughts on Boundaries

I'm beginning to think that off-object observations can provide very rich information than I previously thought. Let me think more about boundaries and their potential implications. (I think I'm beginning to sound like an LLM...)

Boundaries could be a very efficient way to build 3D representations - I presume that mice/rats' whiskers are detecting boundaries to build such object models. A policy that could "trace boundaries" could be an efficient (e.g. fast converging) way to build a morphological model, and possibly take bigger steps within the boundary to efficiently "fill in the holes". Boundaries can also be a sparse representation of our pointcloud object - rather than ~2,000 points, the points along the boundaries of the pointcloud could be a data-efficient way to represent without using deep learning methods. I'm also thinking of "morphological models", e.g. in Excalidraw we have sometimes drawn objects in wireframe.

A "naive" way to incorporate this information is to add a property to nodes of our object model on whether it is boundary node or not, e.g.
```python
@dataclass
class BoundaryNode:
    """Boundary-specific properties for a graph node."""
    is_boundary: bool = False
    boundary_confidence: float = 0.0  # How sure we are this is a boundary
    boundary_normal: Optional[np.ndarray] = None  # Outward-facing normal
    # and maybe more
```
Perhaps these boundary nodes can be connected based on object_id so we can have multiple boundaries needed for compositional objects. 

I think I have gone off on lateral thinking from the original question, but landed on a pretty exciting idea for now. Let's see if I still think this is an exciting approach in days to come. 
