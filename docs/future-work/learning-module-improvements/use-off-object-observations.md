---
title: Use Off-Object Observations
---

There are a variety of instances where a Monty system can find itself perceiving empty space, rather than an object. For example, this can occur either due to a model-free driven action like a saccade moving off the object, or a model-based action like the hypothesis-testing "jump" moving to a location where the object won't be seen.

Currently we have methods to then move the sensor back on to the object, however we do not make use of the information that the object was absent at the visualized location. However, this is valuable information, as the absence of the object at a location will be consistent with some object and pose hypotheses, but not others.

Thus, we would like to update the LM's integration of evidence to account for these observations.