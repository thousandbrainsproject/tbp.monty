---
title: Attention System
---

# Attention System
Attention is the process of selecting which parts of the sensory world deserve the next action, and which parts should be ignored for the time being. Many signals can contribute to that selection: the bottom-up salience of a stimulus, the goals of the task at hand, a tendency not to revisit what was just examined, and the boundaries of the object currently being looked at. In Monty, attention is defined by a region of space. Components (e.g., sensor modules) "propose" regions that they consider meaningful. The attention system accumulates and integrates these proposed regions and uses them to filter out goals that do not lie within the attentional region.

The attention system sits between the components that propose goals and the [motor system](motor-system.md) that acts on them. On every step, the Monty class collects goals from its learning modules and sensor modules, collects attention regions from its sensor modules, and passes both to the attention system. The attention system returns the goals that the motor system is allowed to consider. If no attention system is configured, Monty uses a `NoopAttentionSystem` and every goal passes through unchanged.

`DefaultAttentionSystem` works on a sparse voxel grid over the body-centered 3D space. Each step proceeds as follows:

1. The proposed attention regions are voxelized into a new grid. Where several locations land in the same voxel, their weights are pooled.
2. The weights in the existing grid decay toward zero, and voxels whose weights are close enough to zero expire and are removed.
3. The new grid is merged into the existing grid.
4. The proposed goals are filtered against the merged grid, and the surviving goals are returned.

Each of these stages is a configurable component, described below.

## AttentionRegion
An `AttentionRegion` is a set of 3D locations relative to the body, each carrying an attention weight in the range -1 to 1. Positive weights mark places that should attract action, and negative weights mark places that should be inhibited. Regions are currently proposed by sensor modules; see the [Salience Sensor Module](sensor-module/salience-sm.md#segmentationstrategy) for the one that does so today. `AttentionRegion` is defined alongside `Goal` and `Message` in the [cortical messaging protocol module](https://github.com/thousandbrainsproject/tbp.monty/blob/main/src/tbp/monty/cmp.py), but it is not itself a CMP message: it describes a spatial extent rather than a feature at a pose.

## VoxelGrid
The attention system stores its memory in a `VoxelGrid`, a sparse grid of cubic voxels with one weight per occupied voxel. The voxel size is a parameter of the attention system. Voxels are half-open, so a point that lies exactly on the boundary between two voxels belongs to the one with the larger coordinate.

## WeightPooler
When several locations from the proposed regions fall into the same voxel, a `WeightPooler` reduces their weights to a single value. The default, `negative_priority_max_pool`, lets inhibition dominate: if any location in the voxel carries a negative weight, the most negative weight is used, otherwise the maximum weight is used.

## VoxelGridWeightDecay
Before the new proposal is merged in, a `VoxelGridWeightDecay` moves the weights of the existing grid toward zero so that attention fades unless it is renewed. There are two implementations:
 - `NoopDecay`: Leaves every weight untouched, so attention persists until it is overwritten.
 - `LinearWeightDecay`: Moves every weight toward zero by a fixed amount per step, clamping weights that would cross zero. This is the default.

After decay, voxels whose weight magnitude falls below a small tolerance expire and are removed from the grid.

## VoxelGridMerge
A `VoxelGridMerge` combines the decayed grid with the newly proposed grid. The only implementation so far is `Union`: every voxel occupied in either grid is occupied in the result, and where both grids occupy a voxel the newly proposed weight wins.

## GoalFilter
Finally, a `GoalFilter` decides which goals pass to the motor system given the merged grid. There are two implementations:
 - `NoopGoalFilter`: Passes every goal through unchanged.
 - `HardGoalFilter`: Keeps a goal only if its location falls in a voxel with a positive weight. If the grid is empty, all goals pass. If every voxel in the grid is inhibited, goals that fall outside the grid also pass, so that inhibition alone cannot leave the motor system with nothing to do. Goals without a location are never filtered. This is the default.

## Telemetry
`DefaultAttentionSystem` can be configured with an `AttentionSystemTelemetry` that records the proposed grid and the merged grid at every step. The recorded grids are included in Monty's state dictionary and reach the detailed loggers described in [Logging and Analysis](../how-to-use-monty/logging-and-analysis.md).

# Attention System Classes Currently Implemented
| List of all attention system classes | Description |
| --- | --- |
| **NoopAttentionSystem** | Passes every goal through unchanged and keeps no state. This is what Monty uses when no attention system is configured. |
| **DefaultAttentionSystem** | Maintains a decaying voxel grid built from proposed attention regions and filters goals against it, using the configurable components described above. |

To use the default attention system in an experiment, add `/monty/attention_system: default` to the experiment's defaults list. The compositional object experiments under `src/tbp/monty/conf/experiment/` do this.
