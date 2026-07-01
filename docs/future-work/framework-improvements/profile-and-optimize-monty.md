---
title: Profile and Speed up Monty
description: The current implementation of Montys algorithm is suboptimal and can be optimized. This item includes identifying computational bottlenecks and addressing them.
rfc: optional
estimated-scope: unknown
improved-metric: speed
output-type: PR, analysis
skills: python, performance-optimization
contributor: vkakerbeck
status: evergreen
---

We have profiled Monty a while ago and identified several computational bottlenecks. Some of them have been addressed but there are still many places where the Monty algorithm could be optimized. It has also been a long time since Monty was profiled, so repeating this now would be worth while. 

> [!NOTE]
> One of the most significant computational bottlenecks in Monty at the moment is the KD-Tree search, which is tracked in a separate future work item to [find a faster alternative to KT-Tree search](find-faster-alternative-to-kdtree-search.md).

For the profiling and speedup efforts from back then, see the following two videos:

[2023/02 - Speedup Discussions - Part 1 ](https://www.youtube.com/watch?v=JVz0Km98hLo)

[2023/03 - Speedup Discussions - Part 2 ](https://www.youtube.com/watch?v=XSg6UwGti4c)

You can also find some further discussion and ideas in [this thread](https://forum.thousandbrains.org/t/2023-03-speedup-discussions-part-2/348) on our Forum.