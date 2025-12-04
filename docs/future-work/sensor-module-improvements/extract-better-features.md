---
title: Extract Better Features
description: Add more sophisticated feature extraction to sensor modules.
rfc: optional
estimated-scope: medium
improved-metric: noise, accuracy, numsteps
output-type: monty-feature, PR
skills: python, research, computer-vision
contributor: 
status: evergreen
---

> [!NOTE]
> Even though someone is working on this task, does not mean that no one else should start working on this. Extracting better features in the sensor module is quite dependent on the type of sensor it is connected to (e.g. a sensor connected to an RGBD camera would extract different features to one connected to an ultrasound probe) and even for one type of sensor, there are a multitude of features that could be extracted.
> Anyone is invited to play more with extracting different features in the SM and how this affects Monty's performance.

Currently non-morphological features are very simple, such as extracting the RGB or hue value at the center of the sensor patch.

In the short term, we would like to extract richer features, such as using HTM's spatial-pooler or Local Binary Patterns for visual features, or processing depth information within a patch to approximate tactile texture.

In the longer-term, given the "sub-cortical" nature of this sensory processing, we might also consider neural-network based feature extraction, such as shallow convolutional neural networks, however please see [our FAQ on why Monty does not currently use deep learning](../../how-monty-works/faq-monty.md#why-does-monty-not-make-use-of-deep-learning).

Note that regardless of the approach taken, features should be rotation invariant. For example, a textured pattern should be detected regardless of the sensor's orientation, and the representation of that texture should not be affected by the sensor's orientation.