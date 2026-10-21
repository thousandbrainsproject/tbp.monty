---
title: Separate Location and Feature Messages in CMP
description: Redesign the CMP so that locations and features are sent as separate messages, rather than one message with flags that control which parts should be processed by LMs.
rfc: required
estimated-scope: large
improved-metric: compositional
output-type: RFC, PR, monty-feature
skills: python, refactoring, monty-advanced
contributor: 
status: open
---

Today, the Cortical Messaging Protocol (CMP) couples a location and features together on the same message.
Whether the receiving learning module (LM) uses the message at all, and whether it processes the features in it, is controlled by two flags on the message, `pass_message` and `process_features_in_lm`.
[PR 1051](https://github.com/thousandbrainsproject/tbp.monty/pull/1051) introduced these flags and added a location-only LM mode to allow a sensor module (SM) to send up its location even when its features did not change.
The location information and the feature information therefore share the same message type, and the LM has to flag the location-only step at the start of its matching and exploratory steps.

Ideally, we want two kinds of messages that can be sent independently:
- a location message that communicates the sensor location (or displacements, see [Can We Change the CMP to Use Displacements Instead of Locations?](../voting-improvements/can-we-change-the-cmp-to-use-displacements-instead-of-locations.md)) to the LMs.
- a feature message that communicates all the morphological and non-morphological features.

An LM would update its hypotheses on every location message and only process features when a feature message arrives, which is what the flags currently allows us to do.

Separating the messages would make it easier to send features from different input channels at different rates, which is the goal of [Extend Percept Filtering to LM-to-LM Messages](extend-percept-filtering-to-lm-to-lm-messages.md), and would remove the need for flags that tell the receiver which parts of a message to ignore.
