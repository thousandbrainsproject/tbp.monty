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

**Note**: For rotation, we could similarly "snap" to nearest $45\degree$ increment. 

Finally, neither of the above two approaches are mutually exclusive, and may be beneficial to implement a hybrid approach combining the two.