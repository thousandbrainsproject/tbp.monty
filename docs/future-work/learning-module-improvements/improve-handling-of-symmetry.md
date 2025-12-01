---
title: Improve Handling of Symmetry
description: Remember symmetric poses to recognize symmetry faster on subsequent exposures.
rfc: optional
estimated-scope: unknown
improved-metric: pose, numsteps
output-type: prototype, analysis, PR
skills: python, research, monty
contributor: 
status: open
---

LMs currently recognize symmetry by making multiple observations in a row that are all consistent with a set of multiple poses. I.e. if new observations of an object do not eliminate any of a set of poses, then it is likely that these poses are equivalent/symmetric.

To make this more efficient and robust, we might store symmetric poses in long-term memory, updating them over time. In particular:
- Whenever symmetry is detected, the poses associated with the state could be stored for that object.
- Over-time, we can reduce or expand this list of symmetric poses, enabling the LM to establish with reasonable confidence that an object is in a symmetric pose as soon as the hypothesized poses fall within the list.

By developing an established list of symmetric poses, we might also improve voting on such symmetric poses - see [clean up and simplify voting](../voting-improvements/clean-up-and-simplify-voting.md).