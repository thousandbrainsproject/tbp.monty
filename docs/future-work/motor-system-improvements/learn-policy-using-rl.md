---
title: Learn Policy using RL and Simplified Action Space
---

Learning policies through rewards will become important when we begin implementing complex policies that change the state of the world. However, these could also be relevant for inference and learning, for example by learning when to switch policies instead of adhering to a single heuristic like in the curvature following policy.

In general, we envision that we would use slow, deliberate model-based policies to perform a complex new task, such as one that involves coordinating multiple actuators. Initially, the action would always be performed in this slow, model-based manner. However, with each execution of the task, these sequences of movements provide samples for training a model-free policy to efficiently coordinate relevant actuators *in parallel*, and without the expensive sampling cost of model-based policies.

For example, learning to oppose the finger and thumb in order to make a pinch grasp might initially involve moving one digit until it meets the surface of the object or the other digit, and then applying force with the other. Over time, a model-free policy could learn to move both digits together, with this "pinch policy" recruited by top-down goal-states as necessary.

In addition to supporting efficient, parallel execution of actions, learned model-free policies will be important for more refined movements. For example, the movement required to press a very small button or balance an object might be coarsely guided by a model-based policy, but the fine motor control required to do so would be adjusted via a model-free policy.