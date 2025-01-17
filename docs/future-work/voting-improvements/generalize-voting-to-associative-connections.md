---
title: Generalize Voting to Associative Connections
---
Currently, voting relies on all learning modules sharing the same object ID for any given object, as a form of supervised learning signal. Thanks to this, they can vote on this particular ID when communicating with one-another.

However, in the setting of unsupervised learning, the object ID that is associated with any given model is unique to the parent LM. As such, we need to organically learn the mapping between the object IDs that occur together across different LMs, such that voting can function without any supervised learning signal. This is the same issue faced by the brain, where a neural encoding in one cortical column (e.g. an SDR), needs to be associated with the different SDRs found in other cortical columns.

It is also worth noting that being able to use voting within unsupervised settings will enable us to converge faster, offsetting the issue of not knowing whether we have moved to a new object or not. This relates to the fact that [evidence for objects will rapidly decay](../learning-module-improvements/implement-and-test-rapid-evidence-decay-as-form-of-unsupervised-memory-resetting) in order to better support the unsupervised setting.

Initially, such voting would be explored within modality (two different vision-based LMs learning the same object), or across modalities with similar object structures (e.g. the 3D objects of vision and touch). However, this same approach should unlock important properties, such as associating models that may be structurally very different, like the vision-based object of a cow, and the auditory object of "moo" sounds. Furthermore, this should eventually enable associating learned words with grounded objects, laying the foundations for language.

Finally, this challenge relates to [Use Pose for Voting](./use-pose-for-voting.md), where we would like to vote on the poses of objects, since the learned poses are also going to be unique to each LM.