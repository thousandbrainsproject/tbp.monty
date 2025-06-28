# Unsupervised Object ID Association Learning

This tutorial demonstrates how to run and analyze unsupervised object ID association learning experiments in Monty. These experiments test the ability of learning modules to discover correspondences between their object representations without requiring predefined object labels.

## Overview

Unsupervised association learning addresses a fundamental challenge in cross-modal learning: how can different learning modules (e.g., visual and tactile) learn to associate their internal object representations when they use different object IDs for the same physical object?

For example:
- Visual LM might call a cup `visual_object_1`
- Touch LM might call the same cup `touch_object_2`
- Audio LM might call it `audio_object_5`

The association learning system enables these modules to discover that these different IDs refer to the same physical object through co-occurrence patterns, spatial consistency, and temporal relationships.

## Available Experiments

### 1. Simple Cross-Modal Association

Tests basic association learning between two learning modules (e.g., visual and tactile).

```bash
python benchmarks/run.py -e simple_cross_modal_association
```

**What it tests:**
- Co-occurrence detection between different sensory modalities
- Basic spatial and temporal consistency learning
- Association strength calculation

### 2. Multi-Modal Association

Tests association learning across multiple learning modules simultaneously.

```bash
python benchmarks/run.py -e multi_modal_association
```

**What it tests:**
- Complex multi-way associations between 3+ learning modules
- Handling of conflicting association signals
- Scalability of the association learning system

### 3. Association Strategy Comparison

Compares different association learning strategies and parameter settings.

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
- **Spatial Consistency**: How well spatial relationships align across modalities
- **Temporal Patterns**: Regularity and clustering of association events

### Log Analysis

During experiments, you'll see logs like:

```
INFO - Recorded co-occurrence: visual_object_1 <-> touch_lm:touch_object_3 (count: 15, confidence: 0.892)
INFO - Association strength: visual_object_1 -> touch_lm:touch_object_3 = 0.847
INFO - LM visual_lm: 23 total associations, 18 strong, avg strength: 0.756
```

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
INFO - Recorded co-occurrence: visual_object_1 <-> touch_lm:touch_object_3 (count: 15, confidence: 0.892)
INFO - Association strength: visual_object_1 -> touch_lm:touch_object_3 = 0.847
INFO - LM visual_lm: 23 total associations, 18 strong, avg strength: 0.756
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
- Cross-modal consistency metrics
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
- Check for conflicting signals between modalities

**Memory Issues**:
- Reduce `max_association_memory_size` parameter
- Enable association memory pruning
- Use smaller object sets for initial testing

## Next Steps

- Experiment with different sensor modalities
- Test on more complex multi-object scenarios  
- Integrate with hierarchical learning architectures
- Explore temporal sequence learning extensions

For more advanced usage, see the [Custom Applications Tutorial](using-monty-in-a-custom-application.md) and the [API Reference](../../api-reference/).
