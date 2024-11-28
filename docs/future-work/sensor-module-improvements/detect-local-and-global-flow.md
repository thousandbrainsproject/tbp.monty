---
title: Detect Local and Global Flow
---

Our general view is that there are two sources of flow processed by cortical columns. A larger receptive field sensor helps to estimate global flow, where flow here will be particularly pronounced if the whole object is moving, or the sensor itself is moving. A small receptive-field sensor patch corresponds to the channel by which the primary sensory features (e.g. point-normal, color) arrive. If flow is detected here, but not in the more global channel, then it is likely that just part of the object is moving.

Note that flow can be either optical or based on sensed texture changes for a blind surface agent.

Implementing methods so that we can estimate these two sources of flow and pass them to the LM will be an important step towards modeling objects with complex behaviors, as well as accounting for noise in the motor-system's estimates of self-motion.