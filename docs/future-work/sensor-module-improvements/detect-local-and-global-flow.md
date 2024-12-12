---
title: Detect Local and Global Flow
---

Our general view is that there are two sources of flow processed by cortical columns. These should correspond to:
- Local flow: detected in a small receptive field, and indicates that the *object is moving*.
- Global flow: detected in a larger receptive field, and indicates that the *sensor is moving*.
Note however that depending on the receptive field sizes, it may not be possible for a particular learning module to always distinguish these. For example, if an object is larger than the global-flow receptive field, then from that LM's perspective, it cannot distinguish between the object moving and the sensor moving.

Note that flow can be either optical or based on sensed texture changes for a blind surface agent.

Implementing methods so that we can estimate these two sources of flow and pass them to the LM will be an important step towards modeling objects with complex behaviors, as well as accounting for noise in the motor-system's estimates of self-motion.

Eventually, similar techniques might be used to detect "flow" in how low-level LM representations are changing. This could correspond to movements in non-physical spaces, and enable more abstract representations in higher-level LMs. See also [Can We Change the CMP to Use Displacements Instead of Locations?](../voting-improvements/can-we-change-the-cmp-to-use-displacements-instead-of-locations.md)