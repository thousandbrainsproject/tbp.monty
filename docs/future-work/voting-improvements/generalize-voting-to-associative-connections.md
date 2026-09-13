---
title: Generalize Voting to Associative Connections
description: Learn associations between object representations in different LMs instead of requiring them to use the same representation for object ID.
rfc: required
estimated-scope: large
improved-metric: learning
output-type: RFC, prototype, PR, monty-feature
skills: research, python, monty-advanced
contributor: OgnjenX
status: open
---

> [!NOTE]
> An RFC was previously opened on this topic: https://github.com/thousandbrainsproject/tbp.monty/pull/359 , although it has not undergone any active changes for some time. As such, it is recommended that anyone wishing to work on this create a new RFC guided by the below points.

Currently, voting relies on all learning modules sharing the same object ID for any given object, as a form of supervised learning signal. Thanks to this, they can vote on this particular ID when communicating with one-another.

However, in the setting of unsupervised learning, the object ID that is associated with any given model is unique to the parent LM. As such, we need to organically learn the mapping between the object IDs that occur together across different LMs, such that voting can function without any supervised learning signal. This is the same issue faced by the brain, where a neural encoding in one cortical column (e.g. an SDR), needs to be associated with the different SDRs found in other cortical columns.

It is also worth noting that being able to use voting within unsupervised settings will enable us to converge faster, offsetting the issue of not knowing whether we have moved to a new object or not. This relates to the fact that [evidence for objects will rapidly decay](../learning-module-improvements/implement-and-test-rapid-evidence-decay-as-form-of-unsupervised-memory-resetting) in order to better support the unsupervised setting.

Initially, such voting would be explored within modality (two different vision-based LMs learning the same object), or across modalities with similar object structures (e.g. the 3D objects of vision and touch). However, this same approach should unlock important properties, such as associating models that may be structurally very different, like the vision-based object of a cow, and the auditory object of "moo" sounds. Furthermore, this should eventually enable associating learned words with grounded objects, laying the foundations for language.

To begin tackling this item, a suggested approach is given below:

## Approach Breakdown
- We can use "co-occurrence" as a signal for increasing the associative strength between models in different columns. In particular, if two columns are perceiving objects at the same time, and have strong evidence for them, then there is a chance these are the same object. In this case, they increment an associative connection that can support voting.
- Currently, votes are compiled based on `graph_ids` in `send_out_vote`, and also filtered with `graph_ids` when being received and passed to `_update_evidence_with_vote`. 
- Instead, when we compile and receive votes, these should be filtered such that if a receiving model ID has a strong association with another model ID, then the former is the key that unlocks access to the votes from the latter. In this sense, a receiving model in a receiving column will "listen" for votes coming from other models in other columns that pass a minimum association-strength threshold.
- We would need to ensure that when receiving votes, these can come from multiple models, if multiple models surpass the threshold of associative strength.
- Associative connections can be stored by a receiving LM, in that each LM will internally track what external IDs were associated with certain models. Note that it shouldn't be necessary to modify the CMP signals to track associative connections or their strength.
- This way, most of the machinery of voting at inference time remains unchanged - in particular, the way the k-d tree is used to account for spatial information when voting would remain unchanged. The assumption is that as long as an association is strong, the models can talk with one another in the same reference frame. This is reasonable given that they will have learned similar structured models if those associative connections are good.

## Evaluating an Implementation
- For our existing voting benchmarks, learning takes place with only a single object present. In this case, it is trivial to develop the necessary associative connections for voting. This should prove useful as it means that the first step in this work would look similar to a refactor - even though learning of associative voting connections will technically be unsupervised, we should get the exact same performance when evaluating inference on the voting benchmarks (`randrot_noise_10distinctobj_5lms_dist_agent` and `randrot_noise_77obj_5lms_dist_agent`) as we currently do with supervised voting.
- After verifying the above, it will be necessary to create a new experiment that tests the approach more directly. In the real world, objects are almost never observed in total isolation. As such, there is a risk that spurious voting connections may form between objects that are not statistically predictive of one another. An appropriate benchmark that presents multiple objects during learning would be necessary to show that the associative connections are able to forget and prune spurious associations, while still building associations between LMs representing the same object.