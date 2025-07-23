- Start Date: 2025-07-08
- RFC PR: (leave this empty, it will be filled in after RFC is merged)

# Intelligent Resampling in Monty

This is a high-level RFC on intelligence resampling in Monty, considering the below three questions:

1. How can we re-anchor hypotheses to model points for robustness to noise and distortions?
2. How can we implement and test resampling informed by out-of-reference-frame movement?
3. How can we use prediction errors in off-object observations to eliminate hypotheses? 

## How can we re-anchor hypotheses to model points for robustness to noise and distortions?

"Re-anchoring" means updating the `locations` or `poses` of the `Hypotheses` object to **an existing point** in the object model. The re-anchoring mechanism should be informed by feature observations. 

An example of "distortion" is like the bent TBP logo vs. "standard" TBP logo. In the case of a distortion, a feature / object part / morphology is present at a different location and rotation than would be expected according to the original model. (TODO: include image for clarity and reference for readers). By noise, we refer to errors in location estimates arising from imperfect path integration (TODO: also include image). For example, if relying on optic flow, proprioception, or inertial measurement units, then all of these will result in imperfect estimates of movement displacement and direction.

### Current Status and Objective

Below is the `Hypotheses` class implemented in `tbp.monty==0.8.0`. 
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

Currently, when a hypothesis is initialized, its `poses` attribute is fixed and does not change. While we can create hypotheses with different poses, these new hypotheses will start from zero evidence and will take time to accumulate evidence. So, _we want to implement a mechanism to update the location or pose of a hypothesis without affecting the evidence value_.

### Matching Features 

At a high level, re-anchoring a hypothesis will involve trying to match the **observed features** to **stored features** in object models. If a unique match is found, then we may simply update the location and pose of the hypothesis to that stored in the object model, e.g.

```python
# updating the i-th hypothesis
hypothesis.locations[i] = object_model_location 
hypothesis.poses[i] = object_model_pose
```

Below we work through some questions and implications:

- **How can we meaningfully compare features?**

This depends on specific feature, but for now, we can leverage the existing code in `check_feature_change()` in `FeatureChangeSM` class to compare features. Specifically, as of `tbp.monty==0.8.0`, we have the following definition of distances for `hsv` and `pose_vectors`:

```python
# HSV (line 717)
hue_d = min(abs(current_hue - last_hue), 1 - abs(current_hue - last_hue))

# Pose Vectors (line 729)
angle_between = get_angle(last_feat[0],current_feat[0],)
```

To be considered a "match", _all_ $N$ features should be similar within some threshold. _Why all?_ If, let's say, `hsv` matches but `pose vectors` do not, then this is not informative for an object that has a uniform color all over (e.g. red mug). 

**Note**: Any subsequent extension of observed features in Monty should consider a meaningful distance metric. Note that the above two are examples where L2 distance is not suitable.  

- **What if there are multiple matches?**

This means that the observed set of features exists in multiple locations of the object model, i.e. the observed features are not very distinct. The larger the number of features we are measuring (i.e. larger $N$), the less likely "collisions" are likely to occur. It is also not necessary to re-anchor at every step - we may choose not to re-anchor if there are multiple matches, and only re-anchor if there is a unique match when the sensor if observing a patch containing a unique feature. 

**Note**: We may wish to "mark" these nodes in graph object model when a match has been made to indicate that it contains a unique feature. These may be potentially important nodes or landmarks we may wish to keep, especially if we want to learn a sparse object model. It may also help with computational complexity (see Question below) of the task.

- **What if there are no matches?**

I think this scenario is possible during inference when we add sensor noise that is greater than some threshold for a match. I think there are a couple of ways to proceed, depending on the evidence score:

1. Case 1: Evidence of the hypothesis is low
Perhaps the hypothesis is uncertain about the object and is actually incorrect, hence leading to no match. 
2. Case 2: Evidence for the hypothesis is high
We may want to temporarily increase the threshold to see if we get a match. The larger we have to increase for a match, the larger we could penalize the evidence. 

The simplest implementation shouldn't worry about handling a "no match" case (i.e. just do not perform re-anchoring), but note it could potentially be used as a tool to decrease evidence scores.

- **Computational Complexity**

Because re-anchoring involves comparing of features, this could be costly as a function of: (1) number of features we are observing (the more features we are observing, the more "distance" calculations we will need to do), and (2) number of nodes in object model (the more number of nodes, the more number of comparisons to make).

Currently, case 1 is not much of a concern as we only have `hsv` and `pose_vectors`, but future implementions should try to consider efficient distance calculations AND prioritizing to compare a subset of features. The intuition for the latter is that for an object like a red mug, there is no real benefit of comparing `hsv` between observed and stored because they will always likely match. 

Case 2 is much more of a concern now until we have a more sparse model. We should first benchmark how long it takes to do comparison against ~2,000 points in an object model. One heuristic we could use is just to compare to points within some $\epsilon$-radius of the location, assuming we are re-anchoring to a nearby point. This may be valid if we frequently re-anchor and update the hypotheses, so we never accumulate large errors. Finally, we could try to prioritize the search by first comparing to nodes that were "marked" as unique. 

- **If our models are sparse, then it might actually throw us off from the correct estimate of the location. How can we mitigate this?**

In a sparse model, the nearest stored point might be significantly distant from the actual location on the object model. There may be a few ways to mitigate this:

1. Re-anchoring within $\epsilon$-radius
Like Case 2 of above, we could constrain re-anchoring within a certain distance threshold to prevent large jumps. 

2. Interpolating
Instead of updating to an existing point in an object model, we could try to re-anchor to an intermediate point between the hypothesis' location and object model's location based on some confidence measure. This measure could come from how well the observed features match the features stored in object model. This would create a virtual anchor point - perhaps not necessarily stored in object model (to keep model sparse), but still reaping the benefits of re-anchoring during inference. 

In both cases above, I think matching only when features are distinctive are even more important in sparse models. 

- **If we are doing unsupervised learning, what happens if we re-anchor?**

There are several implications of re-anchoring during unsupervised learning that we need to be very careful about. Re-anchoring changes where we think we are (which may possibly affect policy or what we decide to learn next), which could lead to missing parts of the object (by skipping areas from re-anchoring). 

Couple of mitigation strategies:
1. I think how often we may wish to re-anchor may be a factor or argument in implementation, and we may need to disable re-anchoring if we are in early stages of exploration until $N$ steps.
2. To increase the confidence that we are re-anchoring correctly, we may wish to **delay** re-anchoring until we have taken multiple steps - i.e. we have a better reason to re-anchor to a point after several observed features match the stored features in object models. I think this may also better reflect what happens in real life - the relative positions of several features or history of experience gives a strong clue for localization, even in the absence of a distinct, unique feature match at one time point. 

## How can we implement and test resampling informed by out-of-reference-frame movement?

TODO: Insert Case 1 from Excalidraw link. 

The goal of this question is to eliminate hypotheses when a hypothesis thinks it has moved off the object's reference frame. I think this step would occur in the code after updating hypotheses, but before sensing in the next step. 

One way to implement this would be to eliminate hypotheses if they are 10% away from the object's boundary. Note that the value 10% is arbitrary, but I have chosen a relative distance (instead of a fixed distance such as 3 cm) to account for the different sizes of an object. 

**Computational Complexity**

I think a naive approach may be finding the nearest point, and determining whether it is some x% away. To minimize the costly operation of finding the nearest point, I think approximating the object model once with a convex hull, then comparing against just points that constitute the convex hull will be much more efficient. 

TODO: Include diagram. 

## How can we use prediction errors in off-object observations to eliminate hypotheses? 

TODO: Insert diagram from Excalidraw.
Figure X. Two cases where prediction errors may arise in Monty. 

### Case 1: Hypothesis thinks it is within an object, but has actually moved off the object. 

There is also a case where it has moved off an object and has actually landed in another object. This can actually be handled by Case 2, as (presumably) the features sensed by a different object will result in a large prediction error. 

I think to represent off-object observations, it is important to think of **empty/absent features ARE still features**. Below is a Feature class that can handle both on-object and off-object observations.

```python
# some pseudocode
class FeatureSpace:
    """All observations including 'empty'."""
    def __init__(self, base_features: list, include_metadata: bool = True):
        self.base_features = base_features # list of features we are observing/measuring
        self.include_metadata # completely optional but may be helpful in distinguishing whether the feature is from "empty" or off-object or real observation

    def encode_observation(self, raw_observation: dict) -> dict:
        features = {}

        for feat_name in self.base_features:
            if feat_name in raw_observation:
                features.append(raw_observation[feat_name])
            else:
                # Add feature values that represent for 'empty-ness'
                features.append(self._get_missing_value(feat_name))
        
            if self.include_metadata:
                # Information that might capture "emptiness"
                feature_count = ... # number of features from raw_observation
                feature_variance = ... # empty feature will likely have no variance, also maybe including variance for color or pose may or may not be helpful in the future 
                features.append(feature_count, feature_variance)
        return features
    
    def _get_missing_values(self, feat_name: str):
        """Define what 'empty' means for each feature type.

        This can get really tricky depending on the feature, so just take it as an example. An actual PR should very seriously consider these.
        """
        missing_values = {
            "color": [0, 0, 0] # black
            "depth": max_depth # 1 meter?
            # need to think about "empty" for point_normal, curvature, etc.
        }
        return missing_values[feat_name]
```

Thinking through Vision and Touch:
1. Vision: While there is no such thing as truly "off object" (e.g. we will always likely detect some background color and far away depth), it will not have surface normals and principal curvature. Hence, there will be some partially null features. 
2. Touch:  

Depending on the modality, we may not sense everything in the CMP that LMs digest. 

### Case 2: Hypothesis has stayed within the object, but the expected features are incorrect due to slight location or pose mismatch

Between the two cases, this is the far more common scenario. 


### Implications to FeatureChangeSM and Evidence Scores

#### FeatureChangeSM
With the above `class FeatureSpace`:
- Changes in features from on-object to on-object will still work as existing.  
- Off-object to on-object (and vice versa) will be a significant change, and LM will receive "empty" features if moving from on-object to off-object.
- Off-object to off-object may have feature changes, depending on how we define "empty" values. 
- The off-object observation should trigger FeatureChangeSM but _**should not create new nodes in the object model**_.

Changes that would be made to `sensor_module.py`:

1. Handling of `def check_feature_change()` in `sensor_modules.py`

Current `check_feature_change()`:
```python
if not observed_features.get_on_object():
    # Even for the surface-agent sensor, do not return a feature for LM
    # processing that is not on the object
    logger.debug(f"No new point because not on object")
    return False
```

This part will very likely need to be updated to handle off-object observations. 

2. The `use_state` flag

**Note**: The `use_state` flag "is a bool indicating whether the input is 'interesting'" which indicates that it merits processing by the learning module.

For simplicity, we could set `use_state` for off-object observations to be `True` so the observation information can be used. A more sophisticated method can be used if we don't want _all_ off-object observations to be sent to LM - to be thought more during actual implementation. 

#### EvidenceGraphLM / Evidence Updating Mechanism

We will need to add some logic to handle off-object observations, e.g.
```python
def process_observation(self, state):
    if state.is_off_object():  # Detected via feature signature
        # Use for hypothesis elimination
        self.eliminate_hypotheses_at_location(state.location)
        # Or lower evidence 
    else:
        # Normal on-object processing
        self.update_evidence(state)
```

1. The `_calculate_evidence_for_new_locations()` method

Existing mechanism already penalizes mismatches - however, for off-objects observations, we may want to always give large negative evidence regardless of the magnitude of difference (or in case the difference between on-object and off-object are not large enough).

This will naturally decrease evidence score for hypotheses predicting "on-object" at locations near where we observe off-observations! :)

2. When we're learning an object, we don't have a complete graph model. How do we deal with this?

Prediction errors can be a signal for both removing a hypothesis, as well as needing to update a model. We want to think about when to do which, or how to do both (to a degree) at the same time.
Integrating the learning part over time / many observations, is likely going to be part of it.
