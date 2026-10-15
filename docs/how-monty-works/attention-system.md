---
title: Attention System
---

# Attention System
When learning or inferring compositional objects or objects embedded in a larger scene, it is necessary to constrain observations to within the object of interest. A random walk or even salience-driven policy  generally does not keep Monty on an object long enough for it to be inferred. If learning, stray observations that span multiple objects would result in higher-level models rarely receiving information about inferred objects, preventing compositional representations. To address this, Monty can be configured with an attention system.

In Monty, attention is defined by a region of space. Other components (e.g., sensor modules) may "propose" regions that they consider meaningful, and the attention system will fold those proposals into a persistent representation with which goals (which determine actions) may be filtered out.

The attention system sits between the components that propose goals and the [motor system](motor-system.md) that acts on them. On every step, the Monty class collects goals and proposed regions from its sensor modules and learning modules and passes this data onto the attention system, returning a filtered set of goals.

Monty can currently be configured with either the `NoopAttentionSystem` (which does not filter goals at all) or the `DefaultAttentionSystem`. The `DefaultAttentionSystem` maintains a sparse voxel grid in shared, body-centric 3D space. Since the `DefaultAttentionSystem` is Monty's primary implementation, the remainder of this document refers explicitly to its implementation details.

At each step, the `DefaultAttentionSystem` proceeds as follows:

1. The proposed attention regions are voxelized into a new grid. Where several locations land in the same voxel, their weights are pooled.
2. The weights in the existing grid decay toward zero, and voxels whose weights are close enough to zero expire and are removed.
3. The new grid is merged into the existing grid.
4. The proposed goals are filtered against the merged grid, and the surviving goals are returned.

Each of these stages is a configurable component, described below.

## AttentionRegion
An `AttentionRegion` is a set of 3D locations relative to the body, each associated with an "attention weight" Positive weights mark places that should attract action, and negative weights mark places that should be inhibited. Regions are currently proposed by `SalienceSM`'s [segmentation strategy](sensor-module/salience-sm.md#segmentationstrategy), though any sensor module may be written to propose them. Note that `AttentionRegion` is defined alongside `Goal` and `Message` in the [cortical messaging protocol module](https://github.com/thousandbrainsproject/tbp.monty/blob/main/src/tbp/monty/cmp.py).

## VoxelGrid
The `DefaultAttentionSystem` represents an attentional space as a `VoxelGrid`, a sparse grid of cubic voxels with one weight per occupied voxel. The spatial extent of each voxel is configurable via the `DefaultAttentionSystem`'s `voxel_size` parameter. (Note that voxel-based representations are not a requirement of attention system implementations in general -- so long as an implementation obeys the `AttentionSystem` protocol, spaces/volumes  can be represented in any way.)

## WeightPooler
When several locations from the proposed regions fall into the same voxel, their weights must somehow be combined in order to assign their resulting voxel a single weight. This is the job of the `DefaultAttentionSystem`'s `WeightPooler`. By default, we use the `negative_priority_max_pool` which lets inhibition dominate: if any location in the voxel carries a negative weight, the most negative weight is used, otherwise the maximum weight is used.

## VoxelGridWeightDecay
Before the newly proposed grid is merged into the attention system's , a `VoxelGridWeightDecay` moves the weights of the existing grid toward zero so that attention fades unless it is renewed. There are two implementations:
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
