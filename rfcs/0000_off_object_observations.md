- Start Date: 2025-08-12
- RFC PR: (leave this empty, it will be filled in after RFC is merged)

# Sending and Processing Off-Object Observations in Monty

**Note**: This RFC is a spawn from [Question 3 of the Intelligent Resampling in Monty PR (closed)](https://github.com/thousandbrainsproject/tbp.monty/pull/366). For richer theoretical discussions please check the previous RFC. This RFC will put more emphasis on implementation. 

## Question and Goal

The question that we are addressing is: How can we send and process off-object observations in Monty?

The goal of the RFC is to layout current state of Monty and implementation plan to process off-object observations. 

## Summary

This RFC proposes extending Monty's sensory processing pipeline to handle off-object observations (when a sensor has moved to a location in the environment where no features are sensed). Currently, these observations are filtered out and not passed to Learning Modules (LMs), limiting our ability to leverage prediction errors for penalizing hypotheses' evidence. By defining "null features" to represent the absence of morphological features, modifying the Sensor Module (SM) to forward these observations, and updating the LM to process off-object observations, we can enable more intelligent hypothesis management when sensors transition from on-object to off-object status.

![Off-object observation](0000_off_object_observations/off_object_observation.png)

_Figure 1_: Illustration of an off-object observation sensed by the Sensor Module (right). In this RFC, we will utilize this to penalize the evidence of hypotheses that incorrectly think they are still within the object's reference frame (left).

## Motivation

In real-world scenarios, sensors frequently move off objects during exploration - whether transitioning between objects, reaching object boundaries, or moving through empty space. These off-object observations contain valuable information that Monty currently does not utilize. While [Question 2 of the Intelligent Resampling RFC](https://github.com/thousandbrainsproject/tbp.monty/pull/366) addresses hypotheses that have moved beyond the internal model's boundaries using path integration, this RFC focuses on a complementary approach: using sensory prediction errors to penalize hypotheses' evidence when the sensor has moved off the object.

The core idea is to leverage the mismatch between predicted and observed features when sensors transition off objects. When a sensor moves off an object but active hypotheses still predict on-object features, this creates a strong prediction error signal that should penalize those incorrect hypotheses. Currently, off-object observations are filtered out and therefore not sent to the Learning Module, preventing the system from using this information.

## Current Architecture and Limitations

### Overview of Current Sensory Processing Pipeline

Monty's sensory processing pipeline filters observations in `HabitatDistantPatchSM` by marking off-object states as `use_state=False`, and `FeatureChangeSM` (when enabled) adds a further gating layer on top of that change-detection logic. Currently, this filtering excludes off-object observations entirely.

### The Off-Object Filtering Problem

Currently, `FeatureChangeSM` filters out observations where `on_object` is False. 

```python
if not observed_features.get_on_object():
    logger.debug(f"No new point because not on object")
    return False
```
[Source for `sensor_modules.py` lines 680-684](https://github.com/thousandbrainsproject/tbp.monty/blob/9677cc918adeca9ae21233d957c0401e84f482ab/src/tbp/monty/frameworks/models/sensor_modules.py#L680-L684)

This means that when a sensor moves off an object into empty space, the resulting off-object observation, which could provide valuable prediction error signals, is discarded before reaching the LM. Consequently, the LM's `step()` method is never called for these off-object observations, preventing any processing or learning from this information.


### Sensor Modality Considerations

The challenge of representing off-object observations varies by sensor type:

- **Touch sensors**: Naturally detect "nothing" when not in contact with surface.
- **Vision sensors**: Always detect something (even if just background), making the definition of "off-object" more complex and context-dependent.

Taking this into consideration, we treat "null features" as the absence of morphological information at the current location.
## Proposed Implementation

The following implementation addresses the limitations identified above by enabling off-object observations to flow through the sensory processing pipeline and be used for hypothesis elimination. The work breaks down into roughly three steps:

### Step 1: Standardize Sensor Output for Off-Object Observations

First, we want the sensor module to tell us explicitly when it has nothing morphological to report. In practice that may mean:

- Keep emitting full `State` objects even when `on_object == 0`.
- Potentially replace empty dictionary with an explicit sentinel (e.g. `morphological_features = None`).
- Potentially retain non-morphological features since modalities like vision may still see useful context even while off-object.

Updates to the `State` class should it clear to the downstream to know exactly how to interpret "null" features.

### Step 2: Treat On/Off Transitions as Significant Change

The goal of this step is to make sure `use_state` variable can account for four transition possibilities:

1. **On-object --> On-object**: Existing delta threshold feature comparison logic 
2. **On-object --> Off-object**: Treated as significant change; LM receives null features for hypothesis elimination
3. **Off-object --> On-object**: Treated as significant change; LM receives actual features for new processing
4. **Off-object --> Off-object**: No change detected; observation filtered out

### Step 3: Update Learning Module to Process Null Features

Once off-object observations reach the learning system, we need to distinguish between hypotheses that correctly predicted the sensor would leave the object and those that insisted we would still be on surface. There are three high-level pieces to cover:

- **Propagate the on/off-object signal**: The CMP message already carries `on_object`. We should keep that flag available when assembling the inputs for the hypothesis displacer (e.g. by ensuring `channel_features["on_object"]` survives any feature-selection step). That gives every downstream component an explicit indicator that no surface geometry was sensed.
- **Short-circuit feature comparisons when nothing was sensed**: If `on_object == 0`, then we should skip the call to `feature_evidence_calculator.calculate`.
- **Adjust hypothesis evidence selectively**: The `_calculate_evidence_for_new_locations` method already builds a distance-based mask showing which hypotheses remained near stored nodes (`mask == False`) versus those that routed the sensor outside the object's reference frame (`mask == True`). During an off-object step we could penalize only the former group (since their predictions were wrong) and reward the latter group (for correctly predicting "off-object observation"). 

Together, these updates to the LM should enable it to use off-object signals to either decrease the evidence values for hypotheses that think they are on_object and increase evidence values for hypotheses that correctly predicted to be out of object's reference frame.

### Boundary Precision and Edge Cases

An implementation challenge arises when sensors move just beyond an object's surface boundary. Consider a scenario where a sensor (e.g., a finger or eye) transitions from being on an object to hovering just off its edge - close enough that some active hypotheses remain positioned at the object's boundary while the sensor receives null observations.

This situation presents a tolerance versus accuracy dilemma. On one hand, hypotheses located at the very edge of an object's surface might seem "close enough" to the sensor's true position that penalizing them for predicting morphological features (when the sensor detects nothing) could be overly harsh. This becomes particularly complex when considering location noise and the inherent imprecision in sensor positioning.

While storing explicit "edge-of-object" markers along graph boundaries may be helpful, it also introduces several complexities:

- **Saccadic versus continuous movement**: For sensors that jump between locations (like eyes performing saccades) rather than moving smoothly across surfaces (like fingers), determining where to place boundary markers becomes ambiguous
- **Boundary definition**: Objects have complex 3D geometries where defining precise edges is non-trivial
- **Storage overhead**: Maintaining edge information for every possible transition point would substantially increase memory requirements

**Note**: This RFC will be closed as of September 2025 as it relates less to enabling compositional objects but more on efficient hypothesis elimination. If re-opened in the future, the Future Work and Open Questions written below should help guide the researcher develop the idea further. 

## Future Work and Open Questions

Please see the [Open Question](https://github.com/hlee9212/tbp.monty/blob/hlee9212/intelligence_resampling_rfc/rfcs/0000_intelligent_resampling.md#when-were-learning-an-object-we-dont-have-a-complete-graph-model-how-do-we-deal-with-this) from [Previous RFC on Intelligent Resampling](https://github.com/hlee9212/tbp.monty/blob/hlee9212/intelligence_resampling_rfc/rfcs/0000_intelligent_resampling.md). 

Some other questions that may be relevant for future investigation are:

1. **Noise robustness**: How should the system handle sensor noise near object boundaries that might cause intermittent on/off-object classifications?
2. **Learning phase considerations**: Should off-object observation processing behave differently during initial object learning versus subsequent recognition tasks?