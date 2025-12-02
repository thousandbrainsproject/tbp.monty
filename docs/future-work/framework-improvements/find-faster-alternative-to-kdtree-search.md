---
title: Find Faster Alternative to KDTree Search
description: KD-Tree search is Monty's biggest speed bottleneck, so optimizing this would be beneficial.
rfc: optional
estimated-scope: unknown
improved-metric: speed
output-type: analysis, PR
skills: python
contributor: collin-allen512
status: paused
---

As discussed in these videos on Monty's speed ([part 1](https://youtu.be/JVz0Km98hLo), [part 2](https://youtu.be/XSg6UwGti4c)), KD-Tree search is the operation that Monty currently spends most time one.

Collin did a comparison of different methods in May 2025 and found that our current method was the best for our use case at the moment. For the full results, see [this thread](https://thousandbrains.discourse.group/t/gpu-acceleration-of-knn-search-update/620). However, there may be other nearest neighbor algorithms that were not tested there and that would do better in our scenario (finding nearest neighbors in 3D space).

Also see [this thread](https://thousandbrains.discourse.group/t/possibility-of-faster-recognition-with-monty/574/4) on Discourse for more information of other speedups and KD-Tree alternatives we explored.