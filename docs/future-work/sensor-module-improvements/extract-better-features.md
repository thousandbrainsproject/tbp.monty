---
title: Extract Better Features
description: Add more sophisticated feature extraction to sensor modules.
rfc: optional
estimated-scope: medium
improved-metric: noise, accuracy, numsteps
output-type: monty-feature, PR
skills: python, research, computer-vision
contributor: tcdubs, ryan9186, roccolep, nleadholm
status: evergreen
---

> [!NOTE]
> Even though someone is working on this task, it does not mean that no one else should start working on this. Extracting better features in the sensor module is quite dependent on the type of sensor it is connected to (e.g. a sensor connected to an RGBD camera would extract different features to one connected to an ultrasound probe) and even for one type of sensor, there are a multitude of features that could be extracted.
> Anyone is invited to play more with extracting different features in the SM and how this affects Monty's performance.

Currently non-morphological features are very simple, such as extracting the RGB or hue value at the center of the sensor patch.

We would like to extract richer features, such as using HTM's spatial-pooler or Local Binary Patterns for visual features, or processing depth information within a patch to approximate tactile texture. Given the "sub-cortical" nature of this sensory processing, we might also consider neural-network-based feature extraction, such as shallow convolutional neural networks; however, please see [our FAQ on why Monty does not currently use deep learning](../../how-monty-works/faq-monty.md#why-does-monty-not-make-use-of-deep-learning).

### Constraints

Note that regardless of the approach taken, feature extraction should satisfy a set of important constraints. These are generally phrased in the context of recognizing a texture given an RGB input:
1. Sampling/instance invariance: this is the most basic one; essentially other instances of the same texture pattern should be recognized correctly, as should the texture if it translated slightly with respect to the sensor.
2. Rotation invariance: we cannot assume that a sensor will have a consistent orientation with respect to a texture patch.
3. Scale invariance (somewhat): while it is not realistic to expect a model-free representation of a texture to be fully scale invariant, any extraction method should be robust to smaller scale changes (e.g. after zooming in 25% closer to an image).
4. Lighting: assuming the texture extraction is based on light, and not e.g. the fine-grained depth differences detected by a tactile sensor, then robustness to increases or decreases in lighting is important.
5. Viewpoint and surface distortion: similar to the above, a camera-based setup will often sample a texture patch from a novel viewpoint, which can accentuate distortions caused by curved surfaces on an object. While full invariance to such distortions would likely require a [model-based approach](./surface-sm.md), we hope to observe at least some robustness.
6. Blur (e.g. Gaussian blur): this is not critical, but robustness here would correlate with robustness to a range of other, unspecified forms of noise.
7. Continual learning, or sufficient basis: the feature extraction pipeline needs to be able to output a useful representation for an entirely novel texture (e.g., glossy feathers). This is because Monty itself is robust to continual learning. Note that this might be enabled by a form of true continual learning support, or by ensuring that the representational diversity experienced during learning is sufficient to create a unique representation for novel textures.

### LBP as a First Example

As a first step on this task, [a provisional implementation of LBP feature extraction can be found in this PR](https://github.com/thousandbrainsproject/tbp.monty/pull/1020), which in turn was modeled on [earlier work](https://github.com/thousandbrainsproject/tbp.monty/pull/967). Early results did not show a strong benefit for Monty, suggesting that LBP is not able to address the full list of invariances provided above. The particular issue appears to be with viewpoint invariance, suggesting that LBP may still be useful for tactile sensors that do not have access to light, and which otherwise lack a means of representing fine-grained depth variations on a surface.

### A Texture Classification Benchmark

Alongside implementing a prototype of LBP extraction within Monty, this work led to the development of a [benchmark for evaluating feature extraction algorithms](https://github.com/thousandbrainsproject/tbp.lbp_benchmark). Currently this benchmark only supports LBP-based algorithms, but it could easily be adapted to evaluate others. When starting such work, a promising first task would be to ensure that all of the invariances listed above (notably viewpoint and surface distortion) are reflected in the benchmark evaluations.