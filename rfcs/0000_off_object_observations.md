- Start Date: 2025-08-12
- RFC PR: (leave this empty, it will be filled in after RFC is merged)

# Sending and Processing Off-Object Observations in Monty

**Note**: This RFC is a spawn from [Question 3 of the Intelligent Resampling in Monty PR (closed)](https://github.com/thousandbrainsproject/tbp.monty/pull/366). For richer theoretical discussions please check the previous RFC. This RFC will put more emphasis on implementation. 

## Question and Goal

The question that we are addressing is: How can we send and process off-object observations in Monty?

The goal of the RFC is to layout current state of Monty and implementation plan to process off-object observations. 

## Summary

This RFC proposes extending Monty's sensory processing pipeline to handle off-object observations (when a sensor has moved beyond an object's reference frame). Currently, these observations are filtered out and not passed to Learning Modules (LMs), limiting our ability to leverage prediction errors for penalizing hypotheses' evidence. By defining "null features" to represent the absence of morphological features, modifying the `FeatureChangeSM` to forward these observations, and updating the Learning Module to process off-object observations, we can enable more intelligent hypothesis management when sensors transition from on-object to off-object status. This enhancement will improve Monty's ability to recognize when it has moved off an object and reduce evidence for hypotheses that predict the sensor should still be detecting object features. 

<img src="0000_off_object_observations/off_object_observation.png" alt="Off-object observation" />

_Figure 1_: Illustration of an off-object observation sensed by the Sensor Module. In this RFC, we will utilize this to penalize the evidence of hypotheses that incorrectly think they are still within the object's reference frame.

## Motivation

In real-world scenarios, sensors frequently move off objects during exploration - whether transitioning between objects, reaching object boundaries, or moving through empty space. These off-object observations contain valuable information that Monty currently does not utilize. While [Question 2 of the Intelligent Resampling RFC](https://github.com/thousandbrainsproject/tbp.monty/pull/366) addresses hypotheses that have moved beyond object boundaries using path integration, this RFC focuses on a complementary approach: using sensory prediction errors to penalize hypotheses' evidence when the sensor has moved off the object.

The core idea is to leverage the mismatch between predicted and observed features when sensors transition off objects. When a sensor moves off an object but active hypotheses still predict on-object features, this creates a strong prediction error signal that should penalize those incorrect hypotheses. Currently, off-object observations are filtered out by `FeatureChangeSM` and therefore not sent to the Learning Module, preventing the system from using this information.

## Current Architecture and Limitations

### Overview of Current Sensory Processing Pipeline

Monty's sensory processing pipeline uses `FeatureChangeSM` as a gatekeeper to determine which observations reach Learning Modules (LMs). Currently, this filtering excludes off-object observations entirely.

### The Off-Object Filtering Problem

Currently, `FeatureChangeSM` filters out observations where `on_object` is False. 

```python
# Current filtering behavior in sensor_modules.py (lines 680-684)
if not observed_features.get_on_object():
    logger.debug(f"No new point because not on object")
    return False
```

This means that when a sensor moves off an object into empty space, the resulting off-object observation, which could provide valuable prediction error signals, is discarded before reaching the LM. Consequently, the LM's `step()` method is never called for these off-object observations, preventing any processing or learning from this information.

### Current Data Representation for Off-Object Observations

When sensors move off objects, the current implementation creates empty feature dictionaries:

```python
# Existing data representation in sensor_modules.py (lines 267-286)
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
    morphological_features = {}  # Empty dictionary
```

This empty dictionary approach prevents meaningful comparison with predicted features, as the data structure becomes inconsistent between on-object and off-object states.

### Sensor Modality Considerations

The challenge of representing off-object observations varies by sensor type:

- **Touch sensors**: Naturally detect "nothing" when not in contact with surface.
- **Vision sensors**: Always detect something (even if just background), making the definition of "off-object" more complex and context-dependent.

Taking this into consideration, we define "null features" as the absence of **morphological** features (surface normal and principal curvatures), since depending on the sensing modality, some **non-morphological features** may be present (e.g. background color) even when sensor is off-object.

## Proposed Implementation

The following implementation addresses the limitations identified above by enabling off-object observations to flow through the sensory processing pipeline and be used for hypothesis elimination. The approach involves two main steps: modifying the sensor module's filtering behavior and extending learning modules to process off-object observations.

### Step 1: Update `sensor_modules.py` to Send Null Features

The `sensor_modules.py` file requires modifications to enable off-object observation processing. We need to replace the current empty dictionary approach with consistent null feature representation:

```python
NULL_MORPHOLOGICAL_FEATURES = {
    "pose_vectors": np.array([
        [np.nan, np.nan, np.nan],  
        [np.nan, np.nan, np.nan],  
        [np.nan, np.nan, np.nan],  
    ]),
    "pose_fully_defined": False,
    "principal_curvatures": np.array[[np.nan, np.nan]]
    "gaussian_curvature": np.nan
    "mean_curvature": np.nan
    ...
}

# Existing code...
# Proposed update in line 286 of sensor_modules.py
else: 
    morphological_features = NULL_MORPHOLOGICAL_FEATURES
```

We may also consider removing the use of `invalid_signals`, which is currently used to determine `use_state` in line 305 of `sensor_modules.py` (see code below). The current logic sets `invalid_signals` to `True` if surface normal or principal curvature directions are not defined (see `extract_and_add_features()` method in `sensor_modules.py`), but this check becomes less relevant when we process `NULL_MORPHOLOGICAL_FEATURES` for off-object observations. 

```python
# current implementation around line 305 of sensor_modules.py
observed_state = State(
    location=np.array([x, y, z]),
    morphological_features=morphological_features,
    non_morphological_features=features,
    confidence=1.0,
    use_state=bool(morphological_features["on_object"]) and not invalid_signals, # line 305
    sender_id=self.sensor_module_id,
    sender_type="SM",
)
```

Instead, the `use_state` will depend on which of the four transitions are happening:

1. **On-object --> On-object**: Existing delta threshold feature comparison logic 
2. **On-object --> Off-object**: Treated as significant change; LM receives null features for hypothesis elimination
3. **Off-object --> On-object**: Treated as significant change; LM receives actual features for new processing
4. **Off-object --> Off-object**: No change detected; observation filtered out

#### A. Transition-Aware Logic in `check_feature_change()`

Transform the current "filter-out" approach (lines 680-684) to a "transition-aware" approach:

```python
def check_feature_change(self, observed_features):
    """Check for significant changes including on/off-object transitions."""
    
    if self.last_features is None:  # First observation
        return True
    
    current_on_object = observed_features.get_on_object()
    previous_on_object = self.last_features.get_on_object()
    
    if not current_on_object and not previous_on_object:
        # Off-object to Off-object: no change
        return False
    elif not current_on_object and previous_on_object:
        # On-object to Off-object: significant change (prediction error signal)
        logger.debug("Transition from on-object to off-object detected")
        return True
    elif current_on_object and not previous_on_object:
        # Off-object to On-object: significant change (new features detected)
        logger.debug("Transition from off-object to on-object detected")
        return True
    else:
        # On-object to On-object: use existing feature comparison logic
        return self._check_on_object_feature_changes(observed_features)

def _check_on_object_feature_changes(self, observed_features):
    """Extract existing delta threshold comparison logic (lines 686-740)."""
    # Move existing feature comparison code here
    for feature in self.delta_thresholds.keys():
        # ... existing comparison logic unchanged ...
```

### Step 2: Update Learning Module to Process Null Features

The Learning Module requires some modifications to handle off-object observations for hypothesis elimination. The main changes will be in two places:

#### `DefaultHypothesesDisplacer.displace_hypotheses_and_compute_evidence()` 
```python
def displace_hypotheses_and_compute_evidence(
        self,
        channel_displacement: np.ndarray,
        channel_features: dict,
        evidence_update_threshold: float,
        graph_id: str,
        possible_hypotheses: ChannelHypotheses,
        total_hypotheses_count: int,
    ) -> ChannelHypotheses:
    # Check if this is an off-object obesrvation with null features
    if_off_object = ...

    # For off-object observation, apply evidence penalty 
    if is_off_object:
        evidence_penality = ... # may need to be an argument/parameter
        new_evidence = possible_hypotheses.evidence - evidence_penalty
        return ChannelHypotheses(
            input_channel==possible_hypotheses.input_channel,
            evidence=new_evidence,
            locations=possible_hypotheses.locations, # No location update
            poses=possible_hypotheses.poses
        )

    # Normal processing continues for on-object observations...
    rotated_displacements = possible_hypotheses.poses.dot(channel_displacement)
    search_locations = possible_hypotheses.locations + rotated_displacements
```

#### `DefaultFeatureEvidenceCalculator.calculate()`
```python
@staticmethod
def calculate(
    channel_feature_array: np.ndarray,
    channel_feature_order: list[str],
    channel_feature_weights: dict,
    channel_query_features: dict,
    channel_tolerances: dict,
    input_channel: str,  # noqa: ARG004
) -> np.ndarray:
    # Check if this is an off-object observation with null features 
    is_off_object = ...

    # For off-object observations, return zero evidence for all nodes
    if if_off_object:
        return np.zeros(channel_feature_array.shape[0])
    
    # Existing code...
```

## Benchmarking

Besides adding new or updating existing unit tests for the above functions in `FeatureChangeSM`, `DefaultHypothesesDisplacer`, and `DefaultFeatureEvidenceCalculator` for correctness of implementation, the expectation for results on benchmarks are:

- **Accuracy**: In best case scenarios, implementing this RFC will improve accuracy in object recognition to more quickly eliminate bad hypotheses, but it is not a guarantee. It should not decrease the performance, however.
- **Steps to Convergence**: This is where this RFC will likely have the largest impact. By quickly removing bad hypotheses, Monty may converge faster.


## Future Work

Please see the [Open Question](https://github.com/hlee9212/tbp.monty/blob/hlee9212/intelligence_resampling_rfc/rfcs/0000_intelligent_resampling.md#when-were-learning-an-object-we-dont-have-a-complete-graph-model-how-do-we-deal-with-this) from [Previous RFC on Intelligent Resampling](https://github.com/hlee9212/tbp.monty/blob/hlee9212/intelligence_resampling_rfc/rfcs/0000_intelligent_resampling.md). 