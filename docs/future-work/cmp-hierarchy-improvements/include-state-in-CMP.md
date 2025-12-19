---
title: Include State in CMP
description: Include the inferred state of an object as part of the CMP message.
rfc: optional
estimated-scope: small
improved-metric: dynamic
output-type: PR, monty-feature
skills: python
contributor: 
status: open
---

This item relates to the broader goal of [modeling object behaviors in Monty](../../theory/object-behaviors.md#implementation-in-monty).

To infer object or behavioral state efficiently and communicate it between LMs, we need to add it to the CMP messages. This can be done by updating the CMP definition (currently represented in the unfortunately named `State` class whose name we should probably change in the scope of this).

![The CMP message should contain information about the inferred state of objects in addition to the existing information on model location, orientation, and ID.](../../figures/theory/state_in_CMP.png#width=300px)