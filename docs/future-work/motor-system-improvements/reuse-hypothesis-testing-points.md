---
title: Reuse Hypothesis-Testing Policy Target Points
---

The hypothesis-testing policy is able to generate candidate points on an object that, when observed, should rapidly disambiguate between similar objects, or between similar poses of the same object.

Generating these points requires a model-based policy that simulates the overlap in the graphs between the two most likely objects (or the two most likely poses of the same object). This is a relatively expensive operation, and so one approach would be to store these points in long-term memory, reusing them in future episodes.

For example, when we have first learn about the concept of a mug, we might need to deliberately think about the fact that its handle is what distinguishes it from many other cylindrical objects. However, once we have experienced recognizing mugs a few times, we could quickly recall that testing the handle is a good way to confirm whether we are sensing a mug, or some other object. Related to this, an LM can track how sensing different regions of an object affects its evidence values for the collective hypotheses - those areas that have a disproportionate effect on a top hyptohesis are likely to be good candidates for testing in future episodes.