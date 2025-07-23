- Start Date: 2025-07-08
- RFC PR: (leave this empty, it will be filled in after RFC is merged)

# Intelligent Resampling in Monty

This is a high-level RFC on intelligence resampling in Monty, considering the below three questions:

1. How can we realign hypotheses to model points for robustness to noise and distortions?
2. How can we implement and test resampling informed by out-of-reference-frame movement?
3. How can we use prediction errors in off-object observations to eliminate hypotheses? 

## Note on Terminology

In neuroscience, the term "re-anchoring" may be used broadly. In this RFC, we specify re-anchoring to include both:
1. **Remapping** - when we re-anchor to a new object
2. **Realignment** - when we re-anchor to correct for phase and orientation within a reference frame

This RFC primarily focuses on realignment of hypotheses.

## How can we realign hypotheses to model points for robustness to noise and distortions?

Realigning means updating the `locations` or `poses` of the `Hypotheses` object to an existing point in the object model. This mechanism is informed by feature observations.

**Distortion** refers to cases where features, object parts, or morphologies appear at different locations and rotations than expected in the original model (e.g., a bent TBP logo vs. the standard TBP logo). **Noise** refers to errors in location estimates from imperfect path integration, such as inaccuracies in optic flow, proprioception, or inertial measurement units that lead to imperfect estimates of movement displacement and direction.

### Problem Statement and Proposed Solution

The `Hypotheses` class in `tbp.monty==0.8.0` is defined as follows:

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

**Current limitation:** When a hypothesis is initialized, its `poses` attribute remains fixed. While we can create new hypotheses with different poses, these start from zero evidence and require time to accumulate confidence.

**Objective:** Implement a mechanism to update the location or pose of existing hypotheses without resetting their accumulated evidence values.

### Feature Matching for Hypothesis Realignment

At a high level, realigning a hypothesis involves matching **observed features** to **stored features** in object models. If a unique match is found, we can update the hypothesis location and pose to match the stored object model.

Below we work through some key questions and implications:

#### How can we meaningfully compare features?

The comparison method depends on the specific feature type. For now, we can leverage existing code in the `check_feature_change()` method of the `FeatureChangeSM` class. As of `tbp.monty==0.8.0`, we have the following distance definitions for `hsv` and `pose_vectors`:

```python
# HSV (line 717)
hue_d = min(abs(current_hue - last_hue), 1 - abs(current_hue - last_hue))

# Pose Vectors (line 729)
angle_between = get_angle(last_feat[0],current_feat[0],)
```

For a valid "match", all $N$ features must be similar within specified thresholds. _Why all?_ This requirement is necessary because partial matches can be uninformative, e.g., if `hsv` matches but `pose_vectors` do not, this provides little information for objects with uniform color (e.g., a red mug).

**Note**: Future extensions of observed features in Monty should consider meaningful distance metrics. The examples above demonstrate cases where L2 distance is not suitable.  

#### What if there are multiple matches?

Multiple matches indicate that the observed feature set exists at multiple locations in the object model, i.e. the observed features are not sufficiently distinct. Increasing the number of measured features (larger $N$) reduces the likelihood of such "collisions." It is also not necessary to realign at every step - we may choose not to realign if there are multiple matches, and only realign if there is a unique match.

**Note**: We may want to "mark" nodes in the graph object model when unique matches occur, indicating they contain distinctive features. These landmark nodes could be valuable for learning sparse object models and may improve computational efficiency (see below).

#### What if there are no matches?

This scenario may occur during inference when sensor noise exceeds the matching threshold. There are several approaches depending on the evidence score:

1. **Low evidence hypothesis**: The hypothesis may be uncertain or incorrect, explaining the lack of matches.
2. **High evidence hypothesis**: We could temporarily increase the matching threshold to find a match, penalizing the evidence proportionally to the threshold increase required.

The simplest/initial implementation may consider skipping "no match" cases (i.e., no realignment), though this mechanism could potentially be used to decrease evidence scores.

#### Computational Complexity

Re-anchoring can be computationally expensive due to two factors: (1) the number of observed features (more features require more distance calculations), and (2) the number of nodes in the object model (more nodes require more comparisons).

Currently, case 1 is not a major concern since we only have "few" features (e.g. `rgba` and `pose_vectors`). However, future implementations should consider efficient distance calculations and prioritizing feature subsets. For example, comparing `rgba` for a uniformly colored object (like a red mug) provides little discriminative value.

Case 2 is more concerning until we develop sparser models. We should benchmark comparison times against ~2,000 points in an object model. Potential optimizations include:

- **Local search**: Compare only points within an $\epsilon$-radius of the current location, assuming realignment targets nearby points. This is valid if we realign frequently to prevent large error accumulation.
- **Landmark prioritization**: Prioritize comparisons with nodes previously "marked" as containing unique features. 

#### How can sparse models affect location accuracy?

In sparse models, the nearest stored point might be significantly distant from the actual location. Several mitigation strategies are possible:

1. **Constrained re-anchoring**: Limit re-anchoring to points within an $\epsilon$-radius to prevent large positional jumps.

2. **Interpolation**: Instead of snapping to existing model points, re-anchor to an intermediate position between the hypothesis location and the model location. The interpolation weight could be based on feature matching confidence, which can be proportional to distance error when comparison features. This may create a "virtual anchor point", i.e. a point not necessarily stored in the model (to preserve sparsity) while still benefiting from re-anchoring.

For sparse models, matching only when features are distinctive becomes even more critical.

#### What are the implications for unsupervised learning?

Re-anchoring during unsupervised learning has several important implications that require careful considerationRe-anchoring changes where we think we are 
(which may possibly affect policy or what we decide to learn next), which could lead to missing parts of the object (by skipping areas from re-anchoring). 
- **Policy effects**: Re-anchoring changes our perceived location, potentially affecting exploration policies and learning decisions.
- **Coverage gaps**: Position jumps from re-anchoring could cause the agent to skip regions of the object, leading to incomplete learning. 

Potential mitigation strategies include:

1. **Frequency control**: The re-anchoring frequency should be a configurable parameter. We may need to disable re-anchoring during early exploration phases until sufficient steps have been taken.

2. **Delayed re-anchoring**: To increase confidence in re-anchoring decisions, we should delay re-anchoring until multiple consistent feature matches are observed across several steps. This approach may also better reflects real-world localization, where the relative positions of multiple features and experiential history provide stronger localization cues than a single distinctive feature match. 

## How can we implement and test resampling informed by out-of-reference-frame movement?

The goal of this question is to eliminate hypotheses when a hypothesis thinks it has moved off the object's reference frame. Below depicts the scenario.

![out_of_reference_frame_movement](./0000_intelligent_resampling/out_of_reference_frame_movement.png)
_Figure 1_. Case where hypothesis has moved out of object's reference frame. 

One way to implement this would be to eliminate hypotheses if they are 10% away from the object's boundary. Note that the value 10% is arbitrary, but I have chosen a relative distance (instead of a fixed distance such as 3 cm) to account for the different sizes of an object. Also note I think this step would occur in the code after updating hypotheses, but before sensing in the next step. 

**Computational Complexity**

I think a naive approach may be finding the nearest point, and determining whether it is some x% away. To minimize the costly operation of finding the nearest point, I think approximating the object model once with a convex hull, then comparing against just points that constitute the convex hull will be much more efficient. 

The below is a 2D example to demonstrate the idea. Note that points in object model exists in 3D, but the convex hull idea will still apply. 

![](./0000_intelligent_resampling/convex_hull_example_naive.png) 
![](./0000_intelligent_resampling/convex_hull_example_convex_hull.png)
_Figure 2_. (Left) In naive approach, in order to determine whether the hypothesis is out of an object's reference frame, we may need to compare distance to all points in an object's model, which can currently be around ~2,000 points. (Right) In convex hull approach, we can pre-compute a convex hull (either after training or during pre_epoch in inference) for the stored object model, thus reducing the comparisons to just the points that constitute the object's convex hull. 

Note that while this issue may lessen with sparser models, this will still likely significantly reduce the number of comparisons. 

## How can we use prediction errors in off-object observations to eliminate hypotheses? 

Below shows two types of prediction errors that may arise in Monty. 
![](./0000_intelligent_resampling/predicion_error.png)
_Figure 3_. Two cases where prediction errors may arise in Monty.

### Case 1: Hypothesis thinks it is within an object, but has actually moved off the object. 

**Note**: There is also a case where it has moved off an object and has actually landed in another object. This can actually be handled by Case 2, as (presumably) the features sensed by a different object will result in a large prediction error. 

First, we need a representation of "null" observations in order to compute a prediction error. Here, we will refer to "null" features as absence of **morphological** features (surface normal and principal curvatures), as depending on the modality (vision or touch),  not all **non-morphological features** may be sensed. 

```python
if obs_3d[center_id][3] or (
    not on_object_only and features["object_coverage"] > 0
):
    (
        features,
        morphological_features,
        invalid_signals,
    ) = self.extract_and_add_features(
        features,
        obs_3d,
        rgba_feat,
        depth_feat,
        center_id,
        center_row_col,
        sensor_frame_data,
        world_camera,
    )
else:
    invalid_signals = True
    morphological_features = {}

...

observed_state = State(
    location=np.array([x, y, z]),
    morphological_features=morphological_features,
    non_morphological_features=features, # I think this would throw an error because the variable features would not be defined in off-object case
    confidence=1.0,
    use_state=bool(morphological_features["on_object"]) and not invalid_signals,
    sender_id=self.sensor_module_id,
    sender_type="SM",
)
```

### Case 2: Hypothesis has stayed within the object, but the expected features are incorrect due to slight location or pose mismatch

Between the two cases, this is the far more common scenario. 

### Implications to FeatureChangeSM and Evidence Scores

#### FeatureChangeSM


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

- Changes in features from on-object to on-object will still work as existing.  
- Off-object to on-object (and vice versa) will be a significant change, and LM will receive "empty" features if moving from on-object to off-object.
- Off-object to off-object may have feature changes, depending on how we define "empty" values. 
- The off-object observation should trigger FeatureChangeSM but _**will not create new nodes in the object model**_.

2. The `use_state` flag

**Note**: The `use_state` flag "is a bool indicating whether the input is 'interesting'" which indicates that it merits processing by the learning module.

For simplicity, we could set `use_state` for off-object observations to be `True` so the observation information can be used. A more sophisticated method can be used if we don't want _all_ off-object observations to be sent to LM - to be thought more during actual implementation. 

3. The `on_object` flag

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

### When we're learning an object, we don't have a complete graph model. How do we deal with this?

Prediction errors can be a signal for both removing a hypothesis, as well as needing to update a model. We want to think about when to do which, or how to do both (to a degree) at the same time.
Integrating the learning part over time / many observations, is likely going to be part of it.
