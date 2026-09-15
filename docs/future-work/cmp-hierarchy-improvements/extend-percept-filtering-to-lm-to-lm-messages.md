---
title: Extend Percept Filtering to LM-to-LM Messages
description: Apply percept filters to LM-to-LM messages so a receiving learning module can process lower-level LM features without also receiving sensor module features that step.
rfc: optional
estimated-scope: medium
improved-metric: compositional
output-type: PR, monty-feature
skills: python, monty-advanced
contributor: 
status: open
---

Today, percept filtering is applied only on the output of the sensor modules (SMs), where it gates the sending of SM messages to learning modules (LMs) based on feature change in the SM.
Currently, LMs are only stepped when they receive a sensory input. As a consequence, a higher-level LM (HL-LM) can only process a message received from a lower-level LM (LL-LM) on a step where it also receives features from its own SM.
This rule keeps the compositional graphs that the HL-LM learns for its LM input channels at a similar resolution to the input-channel graph it learns through the SM.

Instead of enforcing that rule on the receiving LM, we want to apply separate percept filters to the LL-LM to HL-LM messages, gating them on feature change in the LL-LM itself.
This decouples LM feature processing from the arrival of SM features while still maintaining the desired, lower-resolution graphs in the HL-LM.

The goal here is to store LM features at an SM location without necessarily receiving SM features on that step.
The first step ([PR 1051](https://github.com/thousandbrainsproject/tbp.monty/pull/1051)) modified the Cortical Messaging Protocol (CMP) so that SMs can send a location-only message to the LMs. This lets us store other input-channel features at the SM locations.
The next step is to let the receiving LM process LM features without receiving SM features.
To do that, we restrict how frequently LM features are sent by applying percept filters on the LM input channel, rather than relying on SM feature filtering, so we can control and maintain the correct graph resolution in the receiving LM.
LM outputs would pass through a percept change filter that updates the `process_features_in_lm` flag, similar to the SM but applied to LMs.
