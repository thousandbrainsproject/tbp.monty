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

In experiments where we have sensor noise, we may wish to "re-anchor" hypotheses to take into account of these distortions. By "re-anchoring", I will assume this means updating `locations` or `poses` attributes of the `Hypotheses` object. 

> [!NOTE] Some Lateral Thoughts
> Another potential way to deal with sensor noise is by adjusting the evidence scores such that it increases proportionally to sensor noise, i.e. evidence scores will accumulate more slowly when there are more noise. However, I do not think that this is a valid approach as we (read: LM) should not have knowledge of how much noise there is, or how much noise is added to sensor modules (SM). 


