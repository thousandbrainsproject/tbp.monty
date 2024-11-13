---
title: Architecture Overview
---
There are three major components that play a role in the architecture: sensors, learning modules, and actuators [\[1\]](#footnote1). These three components are tied together by a common messaging protocol, which we call the cortical messaging protocol (CMP). Due to the unified messaging protocol, the inner workings of each individual component can be quite varied as long as they have the appropriate interfaces [\[2\]](#footnote2).

## Footnotes

<a name="footnote1">1</a>: Sensors may be actuators and could have capabilities to take a motor command to move or attend to a new location.

<a name="footnote2">2</a>: In general, the learning modules in an instance of Monty will adhere to the concepts described herein, however it is possible to augment Monty with alternative learning modules. For example, we do _not_ anticipate that the learning modules described herein will be useful for calculating the result of numerical functions, or for predicting the structure of a protein given its genetic sequence. Alternative systems could therefore be leveraged for such tasks and then interfaced according to the CMP.