---
title: Extract Better Features
---

Currently non-morphological features are very simple, such as extracting the RGB or hue value at the center of the sensor patch.

In the short term, we would like to extract richer features, such as using HTM's spatial-pooler or Local Binary Patterns for visual features, or processing depth information within a patch to approximate tactile texture.

In the longer-term, given the "sub-cortical" nature of this sensory processing, we might also consider neural-network based feature extraction, such as shallow convolutional neural networks, however please see [our FAQ on why Monty does not currently use deep learning](../../how-monty-works/faq-monty.md#why-does-monty-not-make-use-of-deep-learning).