---
title: Explore Alternative Segmentation Strategies
description: SlicMerge is implemented as the first segmentation strategy for SalienceSM. Continue exploring alternatives as the need arises.
rfc: not-required
estimated-scope: medium
improved-metric: compositional, multi-object, real-world
output-type: monty-feature, PR, experiments
skills: python, computer-vision
contributor: scottcanoe, tristanls-tbp
status: evergreen
---

`SalienceSM` can be configured with a `SegmentationStrategy` that marks the pixels around the point of fixation which appear to form a coherent region. The sensor module turns this mask into an `AttentionRegion` so that the [attention system](../../how-monty-works/attention-system.md) can keep the agent on the object it is currently looking at, rather than letting a salient feature elsewhere in the field of view pull the sensor away. See the [SalienceSM documentation](../../how-monty-works/sensor-module/salience-sm.md#segmentationstrategy) for details.

As of this writing, we have implemented one segmentation strategy -- `SlicMerge`. `SlicMerge` segments the image into SLIC superpixels and then grows a region outward from the superpixel under the fixation point by merging neighbors whose mean LAB color is within a threshold. `SlicMerge` was implemented since it performs reasonably well with the logos-on-objects dataset, but it does have some drawbacks:
 - Because it relies on color alone, it can bleed across object boundaries where adjacent objects have similar colors. This is especially problematic when objects don't live in the void. It's very easy for `SlicMerge` to "spill" onto a nearby wall or floor that is visibly behind an object.
 - When attempting to isolate an object that contains edge-heavy content (e.g., a logo with text), `SlicMerge` has difficulty extracting a whole area rather than just letters or sub-objects. While `SlicMerge` does have some knobs that can be turned to encourage larger region extraction, they're generally brittle and don't seem to be a generally useful way to handle this issue.

Directions worth exploring:
- Using three-dimensional surface continuity to inform merging.
- Using texture-level statistics to inform merging.
- Use top-down signals to influence the segmentation strategy as required. For example, biasing a particular region size or feature properties (such as low vs. high frequency/coarse vs. detailed features).
- Eventually we will be working with moving items and/or a smoothly moving agent. In this case, we can explore using motion cues and/or parallax to separate objects that move relative to their surroundings.

