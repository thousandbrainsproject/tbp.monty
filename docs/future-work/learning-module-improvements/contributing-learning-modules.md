---
title: Contributing Learning Modules
---

There is significant scope for custom learning modules in Monty. In particular, learning modules can take a variety of forms, so long as their input and output channels adhere to the Cortical Messaging Protocol, and that they model objects using reference frames. However, exactly how a "reference frame" is implemented is not specified.

Currently, our main approach is to use explicit graphs in Cartesian space, with evidence values accumulated, somewhat analogous to a particle filter. An example of an alternative approach would be using grid-cell modules to model reference frames.

In the future, we will provide further guidance on how custom learning modules can be designed and implemented. If this is something you're currently interested in, please feel free to reach out to us.