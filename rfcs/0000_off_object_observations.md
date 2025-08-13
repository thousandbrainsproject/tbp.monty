- Start Date: 2025-08-12
- RFC PR: (leave this empty, it will be filled in after RFC is merged)

# Sending and Processing Off-Object Observations in Monty

Note: This RFC is a spawn from [Question 3 of the Intelligent Resampling in Monty PR (closed)](https://github.com/thousandbrainsproject/tbp.monty/pull/366). 

The question: How can we send and process off-object observations in Monty?

## Summary

This RFC proposes extending Monty's sensory processing pipeline to handle off-object observations (when a sensor has moved beyond an object's reference frame). Currently, these observations are filtered out and not passed to Learning Modules (LMs), limiting our ability to leverage prediction errors for hypothesis elimination. By defining "null features" to represent the absence of morphological features and modifying the `FeatureChangeSM` to forward these observations, we can enable more intelligent hypothesis management when sensors transition from on-object to off-object states. This enhancement will improve Monty's ability to recognize when it has moved off an object and eliminate incorrect hypotheses that predict the sensor should still be detecting object features. 

## Motivation

In real-world scenarios, sensors frequently move off objects during exploration - whether transitioning between objects, reaching object boundaries, or moving through empty space. These off-object observations contain valuable information that Monty currently does not utilize. While [Question 2 of the Intelligent Resampling RFC]() addresses hypotheses that have moved beyond object boundaries using path integration, this RFC focuses on a complementary approach: using sensory prediction errors to eliminate hypotheses when the sensor has moved off the object.

The core idea is to leverage the mismatch between predicted and observed features when sensors transition off objects. When a sensor moves off an object but active hypotheses still predict on-object features, this creates a strong prediction error signal that should eliminate those incorrect hypotheses. Currently, off-object observations are filtered out by `FeatureChangeSM` and therefore not sent to the Learning Module, preventing the system from using this information.

## Current Architecture and Limitations

### Overview of Current Sensory Processing Pipeline

In Monty's current architecture, sensory observations flow through a processing pipeline where the `FeatureChangeSM` class acts as a gatekeeper, determining which observations reach the Learning Modules (LMs). This filtering mechanism currently excludes off-object observations, creating a significant limitation in our ability to leverage prediction errors for hypothesis management.

### The Off-Object Filtering Problem

The core limitation is that `FeatureChangeSM` filters out observations where `on_object` is False. This filtering occurs in the `check_feature_change()` method, which returns `False` for off-object observations, preventing the LM from receiving them:

```python
# Current filtering behavior in sensor_modules.py
if not observed_features.get_on_object():
    logger.debug(f"No new point because not on object")
    return False
```

This design choice means that when a sensor moves off an object into empty space, the resulting "null" observation—which could provide valuable prediction error signals—is discarded before reaching the LM.

### Current Data Representation for Off-Object States

When sensors move off objects, the current implementation creates empty feature dictionaries:

```python
# Existing behavior in sensor_modules.py
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

- **Touch sensors**: Naturally detect "nothing" when not in contact with surfaces, making null observations conceptually straightforward
- **Vision sensors**: Always detect something (even if just background), making the definition of "off-object" more complex and context-dependent

For this RFC, we focus primarily on touch-like sensors where off-object states represent a clear absence of morphological features (surface normals and principal curvatures).

## Proposed Implementation

The following implementation addresses the limitations identified above by enabling off-object observations to flow through the sensory processing pipeline and be used for hypothesis elimination. The approach involves three main steps: standardizing null feature representation, modifying the sensor module's filtering behavior, and extending learning modules to process off-object observations.

### Step 1: Define Consistent Null Feature Representation

To enable meaningful prediction error calculations, we need to define standardized null values that maintain consistent data structure across on-object and off-object states:

```python
NULL_MORPHOLOGICAL_FEATURES = {
    "pose_vectors": np.array([np.nan, np.nan, np.nan]),
    "principal_curvatures": np.array([np.nan, np.nan]),
    "mean_curvature": np.nan,
    "gaussian_curvature": np.nan,
    "on_object": False,
}
```

This representation ensures that all observations have consistent keys regardless of on/off-object status, enabling proper comparison with predicted features.

### Step 2: Update `FeatureChangeSM` to Send Null Features

Modify the `check_feature_change()` method in `sensor_modules.py` to handle off-object transitions:

```python
# Updated behavior for handling off-object observations
if not observed_features.get_on_object():
    if self.previous_observation_was_on_object():
        # Transition from on-object to off-object - send null features
        return True  # Signal this as a significant change
    else:
        # Off-object to off-object - no change needed
        return False
```

Expected behavior with these changes:
- **On-object to on-object**: Feature changes work as currently implemented
- **On-object to off-object**: Treated as significant changes; LM receives null features
- **Off-object to on-object**: Treated as significant changes; LM receives actual features  
- **Off-object to off-object**: No feature change

Additionally, update the `use_state` flag logic to allow null observations to reach the LM:

```python
# Modified use_state logic
if morphological_features.get("on_object", False):
    use_state = not invalid_signals  # On-object: existing logic
else:
    # Off-object: allow if this represents a transition
    use_state = self.is_transition_observation() and not invalid_signals
```

### Step 3: Update Learning Module to Process Null Features

Extend the Learning Module's observation processing to handle off-object observations for hypothesis elimination:

```python
def process_observation(self, observation):
    """Extended to handle off-object observations for hypothesis elimination."""
    if observation.get("off_object_transition"):
        # Eliminate hypotheses that predicted on-object features
        self.eliminate_hypotheses_with_prediction_errors(observation)
        # Signal to higher-level LMs about component boundary
        self.signal_component_transition(observation)
    else:
        # Existing on-object processing
        self._process_on_object_observation(observation)

def eliminate_hypotheses_with_prediction_errors(self, null_observation):
    """Eliminate hypotheses that predicted features when none were observed."""
    hypotheses_to_remove = []
    
    for hypothesis_id, hypothesis in self.active_hypotheses.items():
        predicted_features = hypothesis.get_predicted_features()
        if predicted_features.get("on_object", False):
            # Hypothesis predicted on-object features but we observed null
            prediction_error = self.calculate_prediction_error(
                predicted_features, null_observation
            )
            if prediction_error > self.elimination_threshold:
                hypotheses_to_remove.append(hypothesis_id)
    
    for hypothesis_id in hypotheses_to_remove:
        self.eliminate_hypothesis(hypothesis_id)
```

## Future Work

### When we're learning an object, we don't have a complete graph model. How do we deal with this?

During object learning, prediction errors present a complex decision point: they may indicate (1) incorrect hypotheses that should be eliminated, (2) an incomplete model that needs updating, or (3) both. These interpretations are not mutually exclusive - a large prediction error might mean we should eliminate current hypotheses AND learn or update our models.

This challenge becomes particularly complex when learning and inference are interleaved. In evaluation mode (no model updates), we may use prediction errors to eliminate hypotheses assuming the object model is considered complete. In learning mode, prediction errors may indicate model needs updating since the model is incomplete.

The key challenge is deciding whether to:
- Learn an entirely new object model (when encountering a truly novel object)
- Update an existing model; this might be necessary when encountering a known object in a new setting (e.g. different lighting), if the model is incomplete (e.g. unexplored part), or it has changed (e.g., your favorite mug now has a chip in it)
- Simply eliminate incorrect hypotheses (when the models are sufficient but we're at the wrong location/on a different object)

To deal with continuous learning and refinement of potentially incomplete models, we can utilize metadata stored in object models as a proxy/heuristic to use prediction errors in one way or another. Heuristics may be:

- **Hypothesis coverage**: If some hypotheses remain valid after prediction errors, the existing models are likely sufficient and we should focus on hypothesis elimination. However, if all hypotheses are eliminated (or less than a certain percentage threshold), we may need to learn a new model or update an existing one.

- **Observation frequency** (e.g., `_observation_count` in `GridObjectModel`): Frequently visited locations with high observation counts suggest the model is well-learned at that location, biasing toward hypothesis elimination rather than model updates.

- **Compositionality**: When prediction errors occur at specific locations on an otherwise well-matched familiar object, consider learning the modification as a compositional element rather than updating the base model. This preserves the integrity of well-learned models while capturing local variations or modifications.

- **Error magnitude thresholds**: Very large prediction errors across all features may suggest a novel object requiring a new model. Moderate errors might indicate the need for model updates or hypothesis refinement.

- **Feature-specific patterns**: If morphological features match but non-morphological features (like color) differ significantly, this might indicate object variations / keyframe / need to separate morphological and feature models.