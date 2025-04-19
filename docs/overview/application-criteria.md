---
title: Application Criteria
description: What can you use Monty for?
---
# Requirements
**Monty is designed for sensorimotor applications.** It is **not designed to learn from static datasets** like many current AI systems are, and so it may not be a drop-in replacement for an existing AI use case you have. Our system is made to learn and interact using similar principles to the human brain, the most intelligent and flexible system known to exist. The brain is a sensorimotor system that constantly receives movement input and produces movement outputs. In fact, the only outputs from the brain are for movement. Our system works the same way. It **needs to receive information about how the sensors connected to it are moving in space** in order to learn structured models of whatever it is sensing. The inputs are not just a bag of features, but instead features at poses (location + orientation).

# Potential Applications

> ⚠️ This is a Research Project
> Note that Monty is still a research project, not a full-fledged platform that you can just plug into your existing infrastructure. Although we are actively working on making Monty easy to use and are excited about people who want to test our approach in their applications, we want to be clear that we are not offering an out-of-the-box solution at the moment. There are many [capabilities](./vision-of-the-thousand-brains-project/capabilities-of-the-system.md) that are still on our [research roadmap](../future-work/project-roadmap.md) and the [tbp.monty](https://github.com/thousandbrainsproject/tbp.monty) code base is still in major version zero, meaning that our public API is not stable and could change at any time.

Any application where you have **moving sensors** is a potential application for Monty. This could be physical movement of sensors on a robot. It could also be simulated movement such as our [simulations in Habitat](../how-monty-works/environment-agent.md) or the sensor patch cropped out of a larger 2D image in the [Monty meets world](benchmark-experiments#monty-meets-world) experiments. It could also be movement through conceptual space or another non-physical space such as navigating the internet.

Applications where we anticipate Monty to particularly shine are:
- Applications where **little data** to learn from is available
- Applications where **little compute to train** is available
- Applications where **little compute to do inference** is available (like on edge devices)
- Applications where **no supervised data** is available
- Applications where **continual learning and fast adaptation** is required
- Applications where the agent needs to **generalize/extrapolate to new tasks**
- Applications where **interpretability** is important
- Applications where **robustness** is important (to noise but also samples outside of the training distribution)
- Applications where **multimodal integration** is required or **multimodal transfer** (learning with one modality and inferring with another)
- Applications where the agent needs to **solve a wide range of tasks**
- Applications **where humans do well** but current AI does not

# Capabilities

For a list of current and future capabilities, see [Capabilities of the System](./vision-of-the-thousand-brains-project/capabilities-of-the-system.md). For experiments (and their results) measuring the current capabilities of the Monty implementation, see our [Benchmark Experiments](benchmark-experiments.md).
