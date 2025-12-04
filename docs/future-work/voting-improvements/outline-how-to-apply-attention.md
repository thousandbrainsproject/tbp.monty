---
title: Outline How to Apply Attention
description: Outline how we can gate information in Monty's routing (votes and inputs) to apply attention filters to regions of space.
rfc: required
estimated-scope: medium
improved-metric: multi-object, speed
output-type: theory, RFC
skills: research, literature-review, monty
contributor: thousandbrainsproject
status: scoping
---

As we create Monty systems with more LMs, it will become increasingly important to be able to emphasize the representations in certain LMs over others, as a form of "covert" attention. This will complement the current ability to explicitly attend to a point in space through motor actions.

For example in human children, learning new language concepts significantly benefits from shared attention with adults ("Look at the -"). A combination of attending to a point in space (overt attention), alongside narrowing the scope of active representations, is likely to be important for efficient associative learning.

Implementation-wise, this will likely consist of a mixture of top-down feedback and lateral competition.