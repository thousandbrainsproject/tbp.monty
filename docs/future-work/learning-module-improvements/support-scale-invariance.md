---
title: Support Scale Invariance
---

It remains unclear how scale invariance would be implemented at a neural level, although we have discussed the possibility that the frequency of oscillatory activity in neurons is scaled. This could in turn modulate how movements are accounted for during path integration.

Regardless of the precise implementation, it is reasonable to assume that a given learning module will have a range of scales that it is able to represent, adjusting path integration in the reference frame according to the hypothesized scale. This scale invariance would likely have the following properties:
- Heuristics based on low-level sensory input (e.g. inferred distance) that are used to rapidly propose the most probable scales.
- Testing of different scales in parallel, similar to how we test different poses of an object.
- Storing the most commonly experienced scales in long-term memory, using these to preferentially bias initialized hypotheses, related to [Use Better Priors for Hypothesis Initialization](../learning-module-improvements/use-better-hypothesis-priors.md).

These scales would represent a small sub-sampling of all possible scales, similar to how we test a subset of possible rotations, and consistent with the fact that human estimates of scale and rotation are imperfect and tend to align with common values.

For example, if an enormous coffee mug was on display in an art installation, the inferred distance from perceived depth, together with the size of eye movements, could suggest that - whatever the object - features are separated on the scale of meters. This low-level information would inform testing objects on a large scale, enabling recognition of the object (albeit potentially with a small delay). If a mug was seen at a more typical scale, then it would likely be recognized faster, similar to how humans recognize objects in their more typical orientations more quickly.

Thus, infrastructure for testing multiple scales (i.e. adjusted path integration in reference frames), or bottom-up heuristics to estimate scale, would be useful additions to the learning module.

In addition to the above scale invariance within a single LM, we believe that different LMs in the hierarchy will have a preference for different scales, proportional to the receptive field sizes of their direct sensory input. This would serve a complimentary purpose to the above scale invariance, constraining the space of hypotheses that each LM needs to test. For example, low-level LMs might be particularly adapt at reading lettering/text. More generally, one can think of low-level LMs as being well suited to modeling small, detailed objects, while high-level LMs are better at modeling larger, objects at a coarser level of granularity. Once again, this will result in objects that are of typical scales being recognized more quickly.