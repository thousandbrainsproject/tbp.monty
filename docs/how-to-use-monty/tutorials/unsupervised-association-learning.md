# Unsupervised Object ID Association Learning

This tutorial demonstrates how to run and analyze unsupervised object ID association learning experiments in Monty. These experiments test the ability of learning modules to discover correspondences between their object representations without requiring predefined object labels.

## Overview

Unsupervised association learning addresses a fundamental challenge in distributed learning: how can multiple learning modules that observe overlapping parts of the same scene learn to associate their internal object representations when they use different object IDs for the same physical object?

In the current Monty implementation we focus on *single-modality* association. Multiple visual patch learning modules (LMs) observe different regions of the same object through narrow retinal-like sensors. Even though they belong to the same modality, each LM still develops its own set of object IDs. The association learning system enables these LMs to discover that their different IDs refer to the same physical object through co-occurrence patterns, spatial consistency, and temporal relationships.

## Available Experiments

**Note:** Multi-modal (e.g., visual + touch) association experiments are not currently supported in Monty. The previously documented `multi_modal_association` experiment has been removed until the learning module infrastructure supports it.

### 1. Dual Patch Association

Tests basic association learning between two visual patch learning modules that observe different regions of the same objects.

```bash
python benchmarks/run.py -e simple_cross_modal_association
```

**What it tests:**
- Co-occurrence detection between distinct visual patches of the same modality
- Basic spatial and temporal consistency learning
- Association strength calculation

### 2. Association Strategy Comparison

Compares different association learning strategies and parameter settings for dual patch setups.

```bash
python benchmarks/run.py -e association_strategy_comparison
```

**What it tests:**
- Different weighting strategies (spatial vs temporal vs co-occurrence)
- Parameter sensitivity analysis
- Robustness across different scenarios

## Understanding the Results

### Association Metrics

The experiments track several key metrics:

- **Total Associations**: Number of object ID pairs that have been associated
- **Strong Associations**: Associations above a confidence threshold
- **Average Strength**: Mean association strength across all pairs
- **Spatial Consistency**: How well spatial relationships align across visual patches
- **Temporal Patterns**: Regularity and clustering of association events
- **Confidence Score**: Time-weighted confidence for a candidate association, derived directly from CMP vote `confidence` values. These scores gate co-occurrence logging (via `association_threshold`), accumulate with temporal decay, and contribute as one component to the final association strength (alongside co-occurrence, spatial consistency, and temporal recency). They are not the same as the final association strength.

### Log Analysis

During experiments, you'll see logs like:

```
INFO - Recorded co-occurrence: visual_object_1 <-> patch_1_lm:visual_object_3 (count: 15, confidence: 0.892)
INFO - Association strength: visual_object_1 -> patch_1_lm:visual_object_3 = 0.847
INFO - LM patch_0_lm: 23 total associations, 18 strong, avg strength: 0.756
```

## CMP Message Structure for Association Learning

### Enhanced Cortical Messaging Protocol

The unsupervised association learning system extends Monty's Cortical Messaging Protocol (CMP) to include object IDs and association metadata. This enables learning modules to discover object correspondences without predefined labels.

#### Standard CMP vs. Association-Enhanced CMP

**Standard CMP Vote (Before):**
```python
vote = State(
    location=np.array([1.0, 2.0, 3.0]),
    morphological_features={
        "pose_vectors": rotation_matrix,
        "pose_fully_defined": True,
    },
    non_morphological_features=None,  # No object ID information
    confidence=0.85,
    use_state=True,
    sender_id="visual_lm",
    sender_type="LM",
)
```

**Association-Enhanced CMP Vote (After):**
```python
vote = State(
    location=np.array([1.0, 2.0, 3.0]),
    morphological_features={
        "pose_vectors": rotation_matrix,
        "pose_fully_defined": True,
    },
    non_morphological_features={
        "object_id": "visual_object_1",        # LM-specific object ID
        "association_metadata": {              # Additional context
            "temporal_context": 15,
            "num_observations": 10,
            "current_location": [1.0, 2.0, 3.0],
            "total_associations": 23
        }
    },
    confidence=0.85,
    use_state=True,
    sender_id="visual_lm",
    sender_type="LM",
)
```

#### Key CMP Extensions

1. **Object ID Transmission**: Each vote now includes the sender's unique object ID in `non_morphological_features["object_id"]`

2. **Sender Identification**: Use the standard CMP `State.sender_id` field to identify the source LM (no extra field in `non_morphological_features` is needed)

3. **Confidence Values**: Explicit confidence values from CMP help with association strength calculations

4. **Association Metadata**: Rich contextual information supports sophisticated association learning algorithms

#### How Association Learning Uses CMP Data

```python
def receive_votes(self, vote_data):
    """Process CMP-compliant votes for association learning."""
    # Extract association data from CMP messages
    for vote in vote_data:
        for object_id, states in vote.items():
            for state in states:
                # Extract from CMP structure
                other_object_id = state.non_morphological_features["object_id"]
                sender_id = state.sender_id
                confidence = state.confidence

                # Learn associations based on co-occurrence
                if confidence > self.association_threshold:
                    self._record_co_occurrence(
                        my_objects, sender_id, other_object_id, confidence, state
                    )
```

### Backward Compatibility

The CMP extensions maintain full backward compatibility:
- Standard learning modules ignore `non_morphological_features` content
- Association-enabled modules extract additional information when available
- No changes to core `State` class structure required

## Analyzing Results

### Using Built-in Logging and Output

Monty's built-in logging infrastructure automatically captures detailed association learning metrics during experiments. Results are stored in structured directories under `~/tbp/results/monty/projects/`.

#### Experiment Output Structure

```
~/tbp/results/monty/projects/simple_cross_modal_association/
├── logs/                    # Detailed execution logs
├── stats/                   # CSV files with metrics
├── configs/                 # Experiment configuration snapshots
└── checkpoints/            # Model checkpoints (if enabled)
```

#### Key Log Files to Examine

- **Main logs**: Look for association learning progress in the main experiment logs
- **CSV stats**: Quantitative metrics are automatically saved to CSV files
- **Debug logs**: Enable debug logging for detailed association tracking

#### Monitoring Association Learning

During experiments, monitor the logs for messages like:

```
INFO - Recorded co-occurrence: visual_object_1 <-> patch_1_lm:visual_object_3 (count: 15, confidence: 0.892)
INFO - Association strength: visual_object_1 -> patch_1_lm:visual_object_3 = 0.847
INFO - LM patch_0_lm: 23 total associations, 18 strong, avg strength: 0.756
```

### Key Analysis Questions

When analyzing results, consider:

1. **Convergence**: Do associations stabilize over time? Check association strength trends in logs.
2. **Accuracy**: Do learned associations match ground truth object correspondences? Compare association patterns with known object relationships.
3. **Robustness**: How well do associations handle noise and ambiguity? Look for consistent association strengths across episodes.
4. **Efficiency**: How quickly do strong associations form? Monitor co-occurrence counts and confidence scores.

### Advanced Analysis with Wandb (Optional)

If wandb logging is enabled in your experiment configuration, you can view rich visualizations and metrics in the wandb dashboard:

- Association strength trends over time
- Cross-patch consistency metrics
- Spatial and temporal pattern analysis
- Comparative performance across different strategies

## Customizing Experiments

### Modifying Association Parameters

You can customize association learning by modifying parameters in the experiment configs:

```python
# In benchmarks/configs/unsupervised_association_experiments.py
association_params = {
    'location_weight': 0.7,           # Weight for spatial similarity
    'pose_weight': 0.3,               # Weight for pose similarity  
    'temporal_recency_weight': 0.1,   # Weight for temporal recency
    'co_occurrence_weight': 0.5,      # Weight for co-occurrence frequency
    'spatial_consistency_weight': 0.3, # Weight for spatial consistency
    'temporal_consistency_weight': 0.2, # Weight for temporal consistency
}
```

### Adding New Strategies

To test new association strategies:

1. Create new parameter presets in `get_association_params_preset()`
2. Add new experiment configurations
3. Update the experiment names in `benchmarks/configs/names.py`

## Troubleshooting

### Common Issues

**Low Association Strengths**: 
- Check that learning modules are sensing the same objects simultaneously
- Verify spatial and temporal consistency parameters
- Ensure sufficient co-occurrence opportunities

**Slow Convergence**:
- Increase the number of training steps
- Adjust association thresholds
- Check for conflicting signals between visual patches

**Memory Issues**:
- Reduce `max_association_memory_size` parameter
- Enable association memory pruning
- Use smaller object sets for initial testing

## Next Steps

- Experiment with different pairs of visual patches (e.g., varying field-of-view overlap)
- Test on more complex multi-object scenarios  
- Integrate with hierarchical learning architectures
- Explore temporal sequence learning extensions

For more advanced usage, see the [Custom Applications Tutorial](using-monty-in-a-custom-application.md) and the [API Reference](../../api-reference/).
